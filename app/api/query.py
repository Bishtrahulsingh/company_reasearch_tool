from typing import List
# POST /query accepts company and question, calls retrieve()
from fastapi import APIRouter
from app.core.models import Chunk
from app.rag.retriever import retrieve

router = APIRouter(prefix='/api',tags=['query'])

@router.post('/query')
async def make_query(collection_name:str,company:str, query:str)->List[Chunk]:
    chunks = await retrieve(collection_name=collection_name,query=query,company=company)
    return chunks


#GET /companies/{name}/chunks returns all stored chunks for a company
@router.get('/companies/{company}/chunks')
async def get_company_chunks(collection_name:str,company:str)->List[Chunk]:
    chunks = await retrieve_all_chunks(collection_name=collection_name,company=company)
    return chunks