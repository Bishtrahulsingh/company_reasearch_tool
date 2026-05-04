from typing import List, Optional
from fastapi import APIRouter
from app.core.models import Chunk
from app.ingestion.chunking import chunk_text
from app.ingestion.deduplicator import is_duplicate
from app.ingestion.embedding import embed_chunks
from app.ingestion.github_client import get_github_activity
from app.ingestion.qdrant_client import retrieve_all_chunks, upsert_chunks
from app.ingestion.scraper import search_web
from app.rag.retriever import retrieve

router = APIRouter(prefix='/api',tags=['query'])

@router.post('/query')
async def make_query(collection_name:str,company:str, query:str)->List[Chunk]:
    chunks = await retrieve(collection_name=collection_name,query=query,company=company)
    return chunks


COMPANY_QUERY_TEMPLATES = [
    "{company} company overview products services business model customers",
    "{company} founding history CEO leadership team headquarters",
    "{company} revenue funding valuation investors financials",
    "{company} latest news product launches acquisitions partnerships",
    "{company} competitors market position pricing strategy",
]
@router.post('/search/company')
async def search_company(collection_name: str,company: str):
    seen_hashes: set = set()
    all_chunks: List[Chunk] = []

    # github_repo = await find_github_repo(company)
    github_repo=''

    for template in COMPANY_QUERY_TEMPLATES:
        query = template.format(company=company)
        result = await search_web(query=query, company=company)

        for item in result.get('items', []):
            text = item.get('text') or ''
            if not text or is_duplicate(text, seen_hashes):
                continue

            all_chunks.extend(chunk_text(text=text,published_date=item.get('published_at'),company=company,source_url=item.get('url', ''),size=800,overlap=50))

    if github_repo:
        github_prose = await get_github_activity(repo=github_repo, since_days=10)

        if github_prose and not is_duplicate(github_prose, seen_hashes):
            all_chunks.extend(chunk_text(
                text=github_prose,
                published_date=None,
                company=company,
                source_url=f"https://github.com/{github_repo}",
                size=800,
                overlap=50,
            ))

    if not all_chunks:
        return {
            'status': 'no data found',
            'message': f'No content could be extracted for "{company}".'
        }

    embedded_chunks = embed_chunks(all_chunks)
    chunk_dicts = [chunk.model_dump() for chunk in embedded_chunks]
    await upsert_chunks(collection_name=collection_name, chunks=chunk_dicts)

    return {
        'status': 'success',
        'company': company,
        'github_repo': github_repo
    }


@router.get('/companies/{company}/chunks')
async def get_company_chunks(collection_name:str,company:str)->List[Chunk]:
    chunks = await retrieve_all_chunks(collection_name=collection_name,company=company)
    return chunks