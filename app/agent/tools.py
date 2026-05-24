import time
from typing import Any

from pydantic import BaseModel

from app.config.settings import settings
from app.core.dependencies import get_qdrant_client
from app.ingestion.chunking import chunk_text
from app.ingestion.deduplicator import is_duplicate
from app.ingestion.embedding import embed_chunks
from app.ingestion.github_client import get_github_activity
from app.ingestion.qdrant_client import upsert_chunks
from app.ingestion.scraper import search_web
from app.rag.retriever import retrieve

web_search_cost = settings.WEB_SEARCH_COST
COLLECTION_NAME = settings.COLLECTION_NAME

class ToolResult(BaseModel):
    result: Any
    source: str
    latency_ms: float
    cost_usd: float

def _company_key(session_id: str, company: str) -> str:
    return f"{session_id}::{company.lower().strip()}"

async def search_web_tool(query: str, company: str, session_id: str) -> ToolResult:
    start = time.perf_counter()
    key = _company_key(session_id, company)
    result = await search_web(query, company)

    seen_hashes: set = set()
    chunks = []
    for item in result.items:
        text = item.get("text") or ""
        if not text or is_duplicate(text, seen_hashes):
            continue
        chunks.extend(chunk_text(
            text=text,
            published_date=item.get("published_at"),
            company=key,
            source_url=item.get("url", ""),
            size=800,
            overlap=50,
        ))
    if chunks:
        embedded = embed_chunks(chunks)
        client = get_qdrant_client()
        await upsert_chunks(client=client, collection_name=COLLECTION_NAME, chunks=[c.model_dump() for c in embedded])

    latency_ms = (time.perf_counter() - start) * 1000
    return ToolResult(
        result={"summary": result.summary, "stored_chunks": len(chunks)},
        source="serper api",
        latency_ms=round(latency_ms, 2),
        cost_usd=web_search_cost,
    )


async def get_github_tool(repo: str, company: str, session_id: str, since_days: int = 7) -> ToolResult:
    start = time.perf_counter()
    key = _company_key(session_id, company)
    result = await get_github_activity(repo, since_days)

    seen_hashes: set = set()

    if result and not is_duplicate(result, seen_hashes):
        chunks = chunk_text(
            text=result,
            published_date=None,
            company=key,
            source_url=f"https://github.com/{repo}",
            size=800,
            overlap=50,
        )
        embedded = embed_chunks(chunks)
        client = get_qdrant_client()
        await upsert_chunks(client=client, collection_name=COLLECTION_NAME, chunks=[c.model_dump() for c in embedded])

    latency_ms = (time.perf_counter() - start) * 1000
    return ToolResult(
        result=result,
        source=f"github:{repo}",
        latency_ms=round(latency_ms, 2),
        cost_usd=0.0,
    )


async def query_rag_tool(query: str, company: str, session_id: str) -> ToolResult:
    start = time.perf_counter()
    key = _company_key(session_id, company)
    client = get_qdrant_client()
    chunks = await retrieve(client=client, collection_name=COLLECTION_NAME, query=query, company=key)

    latency_ms = (time.perf_counter() - start) * 1000
    return ToolResult(
        result=[c.payload for c in chunks],
        source="qdrant",
        latency_ms=round(latency_ms, 2),
        cost_usd=0.0,
    )


async def delete_company_data(company: str, session_id: str) -> None:
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    key = _company_key(session_id, company)
    client = get_qdrant_client()
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="company", match=MatchValue(value=key))]
        ),
    )