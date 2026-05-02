from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.dependencies import get_qdrant_client
from app.ingestion.deduplicator import is_duplicate
from app.ingestion.github_client import get_github_activity

client = get_qdrant_client()

app = FastAPI()

@app.get('/')
async def index():
    return {'hello': 'world'}


