from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.dependencies import get_qdrant_client
from  app.ingestion.chunking import chunk_text
from app.ingestion.embedding import embed_chunks

client = get_qdrant_client()

app = FastAPI()

@app.get('/')
def index():
    print('hello')
    return {'hello': 'world'}

chunks = chunk_text('helllo my name is rahul singh bisht and i am doing nothing',3,2)
embed_chunks(chunks,2)