import json
import re

from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.settings import settings
from app.agent.graph import _extract_tokens
from app.observability.costlogger import log_generation
from langfuse import get_client, observe, propagate_attributes

if settings.LLM_PROVIDER == 'groq':
    _MODEL_NAME = 'llama-3.1-8b-instant'
else:
    _MODEL_NAME = 'gemini-2.0-flash'

langfuse = get_client()

if settings.LLM_PROVIDER == 'groq':
    llm = ChatGroq(
        model=_MODEL_NAME,
        api_key=settings.GROQ_API_KEY,
        temperature=0
    )
else:
    llm = ChatGoogleGenerativeAI(
        model=_MODEL_NAME,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0
    )


class CriticVerdict(BaseModel):
    approved: bool
    issues: list[str]
    confidence: float


_SYSTEM_PROMPT = """
You are a critic agent
You will receive a question, an answer, and source material
Find problems such as unsourced claims, hallucinated numbers, wrong facts
Return only valid JSON matching this structure: {"approved": true/false, "issues": [], "confidence": 0.0-1.0}
"""


async def run_critic(question:str, answer:str, agent_results:list[dict])-> tuple[CriticVerdict, float]:
    context = ""
    for i, result in enumerate(agent_results):
        agent_answer = result.get('answer', '')[:1000]
        context += f"\nSource {i + 1}:\n{agent_answer}\n"
        context += f"Sources used: {result.get('sources_used', [])}\n"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nAnswer:{answer}\n\nagent_results:\n{context}"}
    ]

    response = await llm.ainvoke(messages)

    input_tokens, output_tokens = _extract_tokens(response)

    cost = log_generation(model=_MODEL_NAME, input_tokens=input_tokens, output_tokens=output_tokens)
    langfuse.update_current_generation(
        model=_MODEL_NAME,
        usage_details={"input_tokens": input_tokens, "output_tokens": output_tokens},
        metadata={"phase": "critic", "step_cost_usd": cost}
    )

    try:
        raw = response.content.strip()
        cleaned = re.sub(r"```json\s*|\s*```", "", raw).strip()
        verdict = CriticVerdict(**json.loads(cleaned))
    except Exception:
        verdict = CriticVerdict(approved=True, issues=[], confidence=0.5)

    return verdict, cost