from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.dependencies import get_qdrant_client
from app.api.query import router as query_router
from app.ingestion.qdrant_client import create_collection

app = FastAPI()

app.include_router(query_router)
@app.get('/')
async def index():
    return {'hello': 'world'}


