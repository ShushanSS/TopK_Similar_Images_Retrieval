from io import BytesIO
from PIL import Image
import clip
import torch
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_clip, preprocess = clip.load('ViT-B/32', device = device)

class Embedder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        backbone = models.resnet50(weights=weights)
        backbone.fc = nn.Identity()
        self.resnet = backbone
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(2048, embedding_dim)
        self._frozen_sections = [self.resnet.conv1, self.resnet.bn1, self.resnet.layer1, self.resnet.layer2]

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.nn.functional.normalize(x, dim=-1)

    def train(self, mode=True):
        """requires_grad=False does not stop BatchNorm running stats from drifting --
        that's controlled by .training, not requires_grad. Verified: without this
        override, all 24 BN layers in the "frozen" sections drift anyway during training."""
        super().train(mode)
        if mode:
            for section in self._frozen_sections:
                for m in section.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        return self

model_6_epochs = Embedder()
optimizer = torch.optim.Adam(model_6_epochs.parameters(), lr = 1e-4)

checkpoint = torch.load('embedder_full_train_epoch_6.pt', map_location=device)
model_6_epochs.load_state_dict(checkpoint['model'])
model_6_epochs.to(device)
model_6_epochs.eval()

# def get_embedding(image_bytes: bytes) -> np.ndarray:
#     image = Image.open(BytesIO(image_bytes))
#     image = preprocess(image).unsqueeze(0).to(device)
#     with torch.no_grad():
#         embedding = model.encode_image(image)
#         embedding /= embedding.norm(dim=-1, keepdim=True)
#     return embedding.cpu().numpy().astype('float32')
eval_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def get_embedding(image_bytes: bytes) -> np.ndarray:
    model_6_epochs.eval()
    image = Image.open(BytesIO(image_bytes))
    image = eval_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model_6_epochs(image.to(device))
    return embedding.cpu().numpy().astype('float32')

def get_text_embedding(text: str) -> np.ndarray:
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = model_clip.encode_text(tokens)
        embedding /= embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().astype('float32')



"""
1.
When a user uploads a file through the API,
 FastAPI gives you raw bytes — just a sequence of numbers.
  PIL.Image.open() expects either a file path or a file-like object 
(something it can "read" from). BytesIO wraps raw bytes in a 
fake file object that PIL can read from, 
without ever saving anything to disk.


2.
preprocess does three things:

Resizes image to 224×224 (what ViT-B/32 expects)
Converts to tensor with values in [0,1]
Normalizes with ImageNet mean/std

After preprocess, the tensor shape is (3, 224, 224) — 3 color channels, 224×224 pixels.
But CLIP's encoder expects a batch of images: (batch_size, 3, 224, 224). You only have one image, so .unsqueeze(0) adds a batch dimension at position 0:
(3, 224, 224) → unsqueeze(0) → (1, 3, 224, 224)


3.
PyTorch tensors and models live either on CPU or GPU. 
They must be on the same device to interact — 
you can't multiply a CPU tensor with a GPU tensor.
 .to(device) moves the tensor to wherever the model lives. 
 If CLIP is on GPU, your image tensor must also go to GPU before passing 
 it in.

 4.
 During training, PyTorch tracks every operation on tensors to build a computation graph — 
 this is what makes backpropagation possible. 
 That tracking uses memory and compute.
At inference time you're not training — you don't need gradients. 
torch.no_grad() tells PyTorch to skip that tracking entirely,
 making inference faster and using less memory.

5.
 CLIP's raw output vector has arbitrary magnitude. 
 Normalization divides by the L2 norm so every embedding 
 lands on a unit sphere (magnitude = 1).
This matters because your similarity metric is dot product 
(what FAISS IndexFlatIP computes).
 For unit vectors, dot product equals cosine similarity. 
 Without normalization, dot product is affected by vector magnitude, 
 not just direction — a longer vector would score higher regardless of actual similarity.

 6.
 .cpu() — if the model ran on GPU, the output 
 tensor lives on GPU memory. NumPy doesn't know about GPU memory —
  it only works with CPU. This moves the tensor back.
.numpy() — converts a PyTorch tensor to a NumPy array. 
FAISS expects NumPy arrays, not PyTorch tensors.
.astype('float32') — FAISS specifically requires 32-bit floats. 
PyTorch sometimes produces float16 or float64 depending on the model. 
This guarantees the right dtype.
""" 

