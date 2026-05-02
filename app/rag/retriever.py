import uuid

from app.core.models import Chunk
from app.ingestion.chunking import chunk_text
from app.ingestion.embedding import embed_chunks
from app.ingestion.qdrant_client import filter_chunks_by_company_name, upsert_chunks
from app.ingestion.scraper import search_web


async def retrieve(collection_name:str,query:str, company:str,score_threshold:float,top_k:int=20):
    # #search the web and get the data
    # summary,items = await search_web(query=query,company=company)
    #
    # #create chunks
    # chunks = []
    # for item in items:
    #     item_chunks = chunk_text(text=item.get('text',''),published_date=item.get('published_date'),company=company,source_url=item.get('url'),size=800,overlap=50)
    #     chunks.extend(item_chunks)
    #
    # #store chunks in qdrant
    # await upsert_chunks(collection_name=collection_name,chunks=chunks)
    #
    #embed query
    query_vector = embed_chunks([Chunk(text=query,id=uuid.uuid4(),company=company)],1)[0].embedding

    #filter chunks by company name for query
    filtered_chunks = await filter_chunks_by_company_name(collection_name=collection_name,query_vector=query_vector,score_threshold=score_threshold,top_k=top_k,company=company)
    return filtered_chunks


