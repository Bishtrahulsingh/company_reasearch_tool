import chunk
import math

from logger import logger
from qdrant_client.conversions.common_types import HnswConfigDiff, SearchParams, PointStruct
from qdrant_client.http.models import VectorsConfig, VectorParams, Distance

from main import client


def create_collection(name)->None:
    if client.collection_exists(name):
        logger.info(f"Collection {name} already exists")
    else:
        client.create_collection(
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

def upsert_chunks(collection_name, chunks,batch:int=32):
    if not client.collection_exists(collection_name):
        logger.error(f'Collection {collection_name} does not exist')
    else:
        total_size = len(chunks)

        for i in range(0,total_size,batch):
            start = i
            end = min(start+batch,total_size)
            client.upsert(
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
        logger.info(f"Upserted {total_size} chunks into {collection_name}")

