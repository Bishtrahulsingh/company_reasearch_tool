from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.dependencies import get_qdrant_client
from  app.ingestion.chunking import chunk_text
from app.ingestion.embedding import embed_chunks
from app.ingestion.scraper import search_web

client = get_qdrant_client()

app = FastAPI()

@app.get('/')
async def index():
    print('hello')
    await search_web('what is the factors that leads to this growth of apple in 2025', 'apple')
    return {'hello': 'world'}


