import uuid
from fastapi import APIRouter
from pydantic import BaseModel
from app.agent.graph import run_agent


router = APIRouter(prefix="/api", tags=["research"])

class ResearchRequest(BaseModel):
    company: str
    question: str


class ResearchResponse(BaseModel):
    answer: str
    sources_used: list[str]
    tool_calls: list
    total_cost_usd: float


@router.post("/research", response_model=ResearchResponse)
async def research(body: ResearchRequest):
    session_id = str(uuid.uuid4())
    result = await run_agent(
        question=body.question,
        company=body.company,
        session_id=session_id,
    )
    return ResearchResponse(
        answer=result["answer"],
        sources_used=result.get("sources_used", []),
        tool_calls=result.get("tool_calls_log", []),
        total_cost_usd=result.get("cost_usd", 0.0),
    )