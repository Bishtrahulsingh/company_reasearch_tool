import asyncio
import logging
from typing import List

from app.config.settings import settings
from app.core.dependencies import get_qdrant_client
from app.core.models import Chunk
from app.ingestion.chunking import chunk_text
from app.ingestion.deduplicator import is_duplicate
from app.ingestion.embedding import embed_chunks
from app.ingestion.github_client import get_github_activity
from app.ingestion.qdrant_client import upsert_chunks
from app.ingestion.scraper import search_web

logger = logging.getLogger(__name__)

COLLECTION_NAME = settings.COLLECTION_NAME

_QUERY_TEMPLATES = [
    "{company} company overview products services business model customers",
    "{company} founding history CEO leadership team headquarters",
    "{company} revenue funding valuation investors financials",
    "{company} latest news product launches acquisitions partnerships",
    "{company} competitors market position pricing strategy",
]


async def _scrape_web(company: str, company_key: str) -> List[Chunk]:
    coroutines = [
        search_web(query=t.format(company=company), company=company)
        for t in _QUERY_TEMPLATES
    ]
    results = await asyncio.gather(*coroutines, return_exceptions=True)

    seen_hashes: set = set()
    chunks: List[Chunk] = []

    for result in results:
        if isinstance(result, Exception):
            logger.warning("pipeline: web search failed – %s", result)
            continue
        for item in result.items:
            text = item.get("text") or ""
            if not text or is_duplicate(text, seen_hashes):
                continue
            chunks.extend(
                chunk_text(
                    text=text,
                    published_date=item.get("published_at"),
                    company=company_key,
                    source_url=item.get("url", ""),
                    size=800,
                    overlap=50,
                )
            )

    return chunks


async def _scrape_github(
    repo: str, company_key: str, since_days: int = 7
) -> List[Chunk]:
    try:
        text = await get_github_activity(repo, since_days)
    except Exception as exc:
        logger.warning("pipeline: github fetch failed for %s – %s", repo, exc)
        return []

    if not text:
        return []

    return chunk_text(
        text=text,
        published_date=None,
        company=company_key,
        source_url=f"https://github.com/{repo}",
        size=800,
        overlap=50,
    )


async def run_pipeline(
    company: str,
    session_id: str,
    github_repo: str | None = None,
    since_days: int = 7,
) -> dict:
    company_key = f"{session_id}::{company.lower().strip()}"
    client = get_qdrant_client()

    tasks = [_scrape_web(company, company_key)]
    if github_repo:
        tasks.append(_scrape_github(github_repo, company_key, since_days))

    results = await asyncio.gather(*tasks)
    all_chunks: List[Chunk] = [chunk for batch in results for chunk in batch]

    if not all_chunks:
        logger.warning("pipeline: no content found for company=%s", company)
        return {"chunks_stored": 0, "sources": []}

    embedded = embed_chunks(all_chunks)
    await upsert_chunks(
        client=client,
        collection_name=COLLECTION_NAME,
        chunks=[c.model_dump() for c in embedded],
    )

    sources = list(
        dict.fromkeys(c.source_url for c in embedded if c.source_url)
    )
    logger.info(
        "pipeline: stored %d chunks for company=%s", len(embedded), company
    )
    return {"chunks_stored": len(embedded), "sources": sources}