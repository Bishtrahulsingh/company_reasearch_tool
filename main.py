from dotenv import load_dotenv

from app.observability.tracer import add_tracing

load_dotenv(override=False)

import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.dependencies import get_qdrant_client
from app.api.query import router as query_router
from app.api.research import router as research_router
from app.ingestion.qdrant_client import create_collection
from app.config.settings import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    client = get_qdrant_client()
    await create_collection(client, settings.COLLECTION_NAME)
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins='*',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_tracing(app)
app.include_router(query_router)
app.include_router(research_router)

@app.get('/')
async def index():
    return {'hello': 'world'}

@app.get('/session/new')
async def new_session():
    return {"session_id": str(uuid.uuid4())}