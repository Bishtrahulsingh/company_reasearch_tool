from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.dependencies import get_qdrant_client
from app.ingestion.deduplicator import is_duplicate
from app.ingestion.github_client import get_github_activity
from app.api.query import router as query_router

client = get_qdrant_client()

app = FastAPI()
app.include_router(query_router)

@app.get('/')
async def index():
    return {'hello': 'world'}


