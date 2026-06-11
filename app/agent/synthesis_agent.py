from langchain_groq import ChatGroq

from app.config.settings import settings
from app.agent.graph import _extract_tokens
from app.observability.costlogger import log_generation
from langfuse import get_client, observe, propagate_attributes

_MODEL_NAME = 'openai/gpt-oss-20b'
langfuse = get_client()

_SYSTEM_PROMPT = """

You are a synthesis agent
You will receive research results from multiple agents as context
Write one clean structured answer
Every claim must have a source cited inline
Do not make up information only use what you are given
If information is missing say so clearly

"""

llm = ChatGroq(
    model=_MODEL_NAME,
    api_key=settings.GROQ_API_KEY,
    temperature=0.1
)


async def run_synthesis(question: str,company: str,agent_results: list[dict],issues: list[str]=[]) -> dict:
    context = ""
    for i, result in enumerate(agent_results):
        context += f"\nSource {i+1}:\n{result.get('answer', '')}\n"
        context += f"Sources used: {result.get('sources_used', [])}\n"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Company: {company}\nQuestion: {question}\n\nResearch results:\n{context}"}
    ]

    if issues:
        messages.append({
            "role": "user",
            "content": f"Previous answer had these issues, please fix them:\n{"\n".join(issues)}"
        })

    response = await llm.ainvoke(messages)

    input_tokens, output_tokens = _extract_tokens(response)

    cost = log_generation(model=_MODEL_NAME, input_tokens=input_tokens, output_tokens=output_tokens)
    langfuse.update_current_generation(
        model=_MODEL_NAME,
        usage_details={"input_tokens": input_tokens, "output_tokens": output_tokens},
        metadata={"phase": "synthesis", "plan_cost_usd": cost}
    )

    return {
        "answer": response.content.strip(),
        "cost_usd": cost,
    }