from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.dependencies import get_qdrant_client
from app.ingestion.deduplicator import is_duplicate

client = get_qdrant_client()

app = FastAPI()

@app.get('/')
async def index():
    print('hello')
    seen_hashes = set()
    is_duplicate("hello",seen_hashes)
    is_duplicate(' hello',seen_hashes)

    return {'hello': 'world'}


