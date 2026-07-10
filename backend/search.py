import faiss
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

index_clip = faiss.read_index(os.path.join(BASE_DIR, 'full_gallery.index'))
indexx = faiss.read_index(os.path.join(BASE_DIR, 'resnet_full_faiss.index'))

gallery_meta_clip = pd.read_csv(os.path.join(BASE_DIR, 'full_metadata.csv'))
gallery = pd.read_csv(os.path.join(BASE_DIR, 'original_metadata_filtered.csv'))

def search(embedding, k=10):
    scores, indices = indexx.search(embedding, k=k)
    results = gallery.iloc[indices[0]][['item_id','image_path']].copy()
    results['score'] = scores[0]
    return results

def search_text(embedding, k=10):
    scores, indices = index_clip.search(embedding, k=k)
    results = gallery_meta_clip.iloc[indices[0]][['item_id','image_path']].copy()
    results['score'] = scores[0]
    return results

