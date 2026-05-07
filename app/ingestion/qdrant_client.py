from typing import List

from logger import logger
from qdrant_client.models import (
    HnswConfigDiff, PointStruct, Filter, FieldCondition, MatchValue,
    VectorParams, Distance, Match, SparseVector, SparseVectorParams,
    Modifier, Prefetch, FusionQuery, Fusion,PayloadSchemaType
)
from app.core.dependencies import get_qdrant_client
from app.core.models import Chunk

client = get_qdrant_client()


async def create_collection(name) -> None:
    if await client.collection_exists(name):
        logger.info(f"Collection {name} already exists")
        return

    await client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=384, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            "bm25": SparseVectorParams(modifier=Modifier.IDF)
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=1000),
    )
    await client.create_payload_index(
        collection_name=name,
        field_name="company",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info(f"Collection {name} created successfully")


async def upsert_chunks(collection_name, chunks, batch: int = 32):
    if not await client.collection_exists(collection_name):
        logger.error(f"Collection {collection_name} does not exist")
        return

    total_size = len(chunks)
    for i in range(0, total_size, batch):
        start, end = i, min(i + batch, total_size)
        await client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=chunk["id"],
                    vector={
                        "dense": chunk["embedding"],
                        "bm25": SparseVector(
                            indices=chunk["sparse_embedding"]["indices"],
                            values=chunk["sparse_embedding"]["values"],
                        ),
                    },
                    payload={
                        "text":           chunk["text"],
                        "company":        chunk["company"],
                        "source_url":     chunk["source_url"],
                        "published_date": chunk["published_date"],
                        "score":          chunk["score"],
                    },
                )
                for chunk in chunks[start:end]
            ],
        )
    logger.info(f"Upserted {total_size} chunks into {collection_name}")


async def filter_chunks_by_company_name(
    collection_name: str,
    company: str,
    query_vector: List[float],
    sparse_vector: dict,
    score_threshold: float = 0.0,
    top_k: int = 20,
):
    company_filter = Filter(
        must=[
            FieldCondition(
                key="company",
                match=MatchValue(value=company.lower().strip()),
            )
        ]
    )

    results = await client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_vector,
                using="dense",
                filter=company_filter,
                limit=top_k * 2,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"],
                ),
                using="bm25",
                filter=company_filter,
                limit=top_k * 2,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    for point in results.points:
        point.payload["score"] = point.score
    return results.points


async def retrieve_all_chunks(collection_name: str, company: str) -> List[Chunk]:
    company = company.lower().strip()
    results, _ = await client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="company", match=Match(value=company))
            ]
        ),
        limit=100,
    )
    return results