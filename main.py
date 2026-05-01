from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.dependencies import get_qdrant_client
from app.ingestion.deduplicator import is_duplicate
from app.ingestion.github_client import get_github_activity

client = get_qdrant_client()

app = FastAPI()

@app.get('/')
async def index():
    print('hello')
    github_data = await get_github_activity('makenotion/notion-sdk-js',27)
    print(github_data)

    return {'hello': 'world'}


