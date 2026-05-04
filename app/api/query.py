from typing import List
from fastapi import APIRouter
from app.core.models import Chunk
from app.ingestion.chunking import chunk_text
from app.ingestion.qdrant_client import retrieve_all_chunks, upsert_chunks
from app.ingestion.scraper import search_web
from app.rag.retriever import retrieve

router = APIRouter(prefix='/api',tags=['query'])

@router.post('/query')
async def make_query(collection_name:str,company:str, query:str)->List[Chunk]:
    chunks = await retrieve(collection_name=collection_name,query=query,company=company)
    return chunks

@router.post('/search/company')
async def search_company(collection_name:str,company:str):
    #create query

    #search the web and get the data
    summary,items = await search_web(query=query,company=company)

    #create chunks
    chunks = []
    for item in items:
        item_chunks = chunk_text(text=item.get('text',''),published_date=item.get('published_date'),company=company,source_url=item.get('url'),size=800,overlap=50)
        chunks.extend(item_chunks)

    #store chunks in qdrant
    await upsert_chunks(collection_name=collection_name,chunks=chunks)

@router.get('/companies/{company}/chunks')
async def get_company_chunks(collection_name:str,company:str)->List[Chunk]:
    chunks = await retrieve_all_chunks(collection_name=collection_name,company=company)
    return chunks