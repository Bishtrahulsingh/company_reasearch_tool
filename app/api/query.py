import asyncio
import uuid
from typing import List
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.core.dependencies import get_qdrant_client
from app.core.models import Chunk, JobStatus
from app.ingestion.chunking import chunk_text
from app.ingestion.deduplicator import is_duplicate
from app.ingestion.embedding import embed_chunks
from app.ingestion.qdrant_client import retrieve_all_chunks, upsert_chunks
from app.ingestion.scraper import search_web
from app.rag.retriever import retrieve
from app.agent.tools import delete_company_data
from app.agent.graph import run_agent

router = APIRouter(prefix='/api', tags=['query'])


@router.post('/agent/query')
async def agent_query(company: str, session_id: str, question: str):
    result = await run_agent(question=question, company=company, session_id=session_id)
    return {"answer": result["answer"], "cost_usd": result["cost_usd"], "steps": result["steps"]}

COMPANY_QUERY_TEMPLATES = [
    "{company} company overview products services business model customers",
    "{company} founding history CEO leadership team headquarters",
    "{company} revenue funding valuation investors financials",
    "{company} latest news product launches acquisitions partnerships",
    "{company} competitors market position pricing strategy",
]

_jobs: dict[str, JobStatus] = {}


def _company_key(session_id: str, company: str) -> str:
    return f"{session_id}::{company.lower().strip()}"


async def _run(job_id: str, collection_name: str, company: str, session_id: str):
    try:
        client = get_qdrant_client()
        key = _company_key(session_id, company)

        search_coroutines = [
            search_web(query=t.format(company=company), company=company)
            for t in COMPANY_QUERY_TEMPLATES
        ]

        _jobs[job_id] = JobStatus(job_id=job_id, status="running", message="Searching web...")
        search_results = await asyncio.gather(*search_coroutines, return_exceptions=True)

        seen_hashes: set = set()
        all_chunks: List[Chunk] = []

        for result in search_results:
            if isinstance(result, Exception):
                continue
            for item in result.items:
                text = item.get('text') or ''
                if not text or is_duplicate(text, seen_hashes):
                    continue
                all_chunks.extend(chunk_text(
                    text=text,
                    published_date=item.get('published_at'),
                    company=key,
                    source_url=item.get('url', ''),
                    size=800,
                    overlap=50,
                ))


        if not all_chunks:
            _jobs[job_id] = JobStatus(job_id=job_id, status="failed", message="No content found")
            return

        _jobs[job_id] = JobStatus(job_id=job_id, status="running", message="Embedding chunks...")
        embedded_chunks = embed_chunks(all_chunks)

        _jobs[job_id] = JobStatus(job_id=job_id, status="running", message="Storing in Qdrant...")
        await upsert_chunks(client=client, collection_name=collection_name, chunks=[c.model_dump() for c in embedded_chunks])

        _jobs[job_id] = JobStatus(job_id=job_id, status="done", message=f"Ingested {len(embedded_chunks)} chunks")
    except Exception as e:
        _jobs[job_id] = JobStatus(job_id=job_id, status="failed", message=str(e))


@router.post('/query')
async def make_query(collection_name: str, company: str, session_id: str, query: str) -> List[Chunk]:
    key = _company_key(session_id, company)
    client = get_qdrant_client()
    chunks = await retrieve(client=client, collection_name=collection_name, query=query, company=key)
    return chunks



@router.post('/ingest/company', status_code=202)
async def search_company(
    collection_name: str,
    company: str,
    session_id: str,
    background_tasks: BackgroundTasks,
):

    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobStatus(job_id=job_id, status="queued", message="Waiting to start")
    background_tasks.add_task(_run, job_id, collection_name, company, session_id)
    return {"job_id": job_id}


@router.get('/jobs/{job_id}')
async def get_job(job_id: str) -> JobStatus:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@router.get('/companies/{company}/chunks')
async def get_company_chunks(collection_name: str, company: str, session_id: str) -> List[Chunk]:
    key = _company_key(session_id, company)
    client = get_qdrant_client()
    chunks = await retrieve_all_chunks(client=client, collection_name=collection_name, company=key)
    return chunks


@router.delete('/session/{session_id}/company/{company}')
async def clear_session(session_id: str, company: str):
    await delete_company_data(company=company, session_id=session_id)
    return {"deleted": True}