from fastapi import FastAPI, UploadFile, File 
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from model import get_embedding, get_text_embedding
from search import search, search_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post('/search')
async def search_endpoint(file: UploadFile = File(...), k: int = 10):
    contents = await file.read()
    embedding = get_embedding(contents)
    res = search(embedding, k)
    return res.to_dict(orient = 'records')

@app.get('/image')
async def get_image(image_path: str):
    return FileResponse(image_path)

@app.get('/search-text')
async def search_text_endpoint(query: str, k: int = 10):
    embedding = get_text_embedding(query)
    results = search_text(embedding, k=k)
    return results.to_dict(orient='records')

    

"""
app = FastAPI()
Creates the FastAPI application instance. Everything — middleware, routes, endpoints — gets attached to this object. When uvicorn runs main:app, 
it's looking for this specific object.

app.add_middleware(CORSMiddleware, ...)
CORS (Cross-Origin Resource Sharing) is a browser security policy. By default, browsers block requests 
from one origin (React app at localhost:3000) to a different origin (my API at localhost:8000). 
They're on different ports, so they count as different origins.
allow_origins=["*"] — accept requests from any origin. Fine for development, in production you'd restrict this to your actual domain.
Without this your React frontend would get blocked and never receive results.

@app.post('/search')
@app.post is a decorator — it registers the function below it as the handler for POST /search. 
When a request hits that route, FastAPI calls this function.

file: UploadFile = File(...) — FastAPI reads the uploaded file from the request automatically. ... means it's required.
k: int = 10 — FastAPI reads k from the query string (/search?k=5) and converts it to an integer. Defaults to 10 if not provided.

contents = await file.read()
Reads the uploaded file as raw bytes. await is needed because this is an async I/O operation
 — reading from the network doesn't happen instantly, so FastAPI doesn't block the whole server waiting for it.

res.to_dict(orient='records')
res is a pandas DataFrame. HTTP responses need to be JSON — not DataFrames. .to_dict(orient='records') converts it to a list of dicts:
python[
  {"item_id": "id_00000080", "image_path": "/mnt/...", "score": 0.87},
  {"item_id": "id_00000123", "image_path": "/mnt/...", "score": 0.82},
  ...
]
FastAPI serializes this to JSON automatically.

@app.get('/image')
GET /image?image_path=/mnt/c/.../front.jpg
image_path: str — FastAPI reads this from the query string automatically.
FileResponse(image_path) — reads the file from disk and streams it back to the 
browser with the correct content-type header (image/jpeg etc.).
 Without this endpoint, the browser has no way to display images stored on your local filesystem.

 

When you run uvicorn main:app from the backend/ folder, 
Python's working directory is wherever you ran that command from. 
If you ever run it from a different directory —
 say your home folder — then faiss.read_index('gallery.index')
  looks for the file relative to THAT directory and fails with a FileNotFoundError.
__file__ is always the absolute path of search.py itself,
 regardless of where you ran uvicorn from. os.path.dirname(os.path.abspath(__file__)) gives you the folder that search.py lives in. Building paths from there means the files are always found no matter where you launched the server from.

"""