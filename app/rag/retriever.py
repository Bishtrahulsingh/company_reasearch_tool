import uuid

from app.core.models import Chunk
from app.ingestion.chunking import chunk_text
from app.ingestion.embedding import embed_chunks,sparse_embedding_model
from app.ingestion.qdrant_client import filter_chunks_by_company_name, upsert_chunks
from app.ingestion.scraper import search_web

async def retrieve(collection_name:str,query:str, company:str,score_threshold:float=0.0,top_k:int=20):
    #embed query
    #dense embedding
    chunk = embed_chunks([Chunk(text=query,id=uuid.uuid4(),company=company)],1)[0]
    query_vector = chunk.embedding

    #sparse embedding
    sp = list(sparse_embedding_model.embed([query]))[0]
    sparse_vector = {
        'indices':sp.indices.tolist(),
        'values':sp.values.tolist()
    }
    #filter chunks by company name for query
    filtered_chunks = await filter_chunks_by_company_name(collection_name=collection_name,query_vector=query_vector,score_threshold=score_threshold,top_k=top_k,company=company,sparse_vector=sparse_vector)
    return filtered_chunks
