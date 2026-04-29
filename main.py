from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.dependencies import get_qdrant_client

client = get_qdrant_client()

app = FastAPI()

@app.get('/')
def index():
    print('hello')
    return {'hello': 'world'}

