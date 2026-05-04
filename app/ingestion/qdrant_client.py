from typing import List, Dict, Any

from logger import logger
from qdrant_client.conversions.common_types import HnswConfigDiff, SearchParams, PointStruct, Filter
from qdrant_client.grpc import FieldCondition
from qdrant_client.http.models import VectorParams, Distance, models, MatchValue, Match

from app.core.models import Chunk
from main import client



async def create_collection(name)->None:
    if await client.collection_exists(name):
        logger.info(f"Collection {name} already exists")
    else:
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            ),
            search_params=SearchParams(hnsw_ef = 128),
            hnsw_config=HnswConfigDiff(
                m = 16,
                ef_construct=100,
                full_scan_threshold=1000,
            )
        )
        logger.info(f"Collection {name} created successfully")

async def upsert_chunks(collection_name, chunks,batch:int=32):
    if not await client.collection_exists(collection_name):
        logger.error(f'Collection {collection_name} does not exist')
    else:
        total_size = len(chunks)

        for i in range(0,total_size,batch):
            start = i
            end = min(start+batch,total_size)
            await client.upsert(
                collection_name=collection_name,
                points = [PointStruct(
                    id = chunk['id'],
                    vector = chunk['embedding'],
                    payload={
                        'text':chunk['text'],
                        'company': chunk['company'],
                        'source_url': chunk['source_url'],
                        'published_date':chunk['published_date'],
                        'score': chunk['score']
                    }
                ) for chunk in chunks[start:end]]
            )
        logger.info(f"upserted {total_size} chunks into {collection_name}")

async def filter_chunks_by_company_name(collection_name:str,company:str,query_vector:List[float],score_threshold:float=0.7,top_k:int=20)->List[Dict[str,Any]]:
    chunks = await client.query_points(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key='company',
                    match=models.MatchValue(value=company.lower().strip())
                )
            ]
        ),
        limit=top_k,
        score_threshold=score_threshold,
    )

    #store score of each chunk watch Chunk schema for more detail
    return chunks


async def retrieve_all_chunks(collection_name:str, company:str)->List[Chunk]:
    company = company.lower().strip()
    results, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="company",
                    match=Match(value=company)
                )
            ]
        ),
        limit=100
    )

    return results