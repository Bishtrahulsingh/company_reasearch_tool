import asyncio
import json
import logging
import re
from typing import Any, Dict

from langchain_groq import ChatGroq
from langfuse import get_client, observe, propagate_attributes

from app.agent.github_agent import run as run_github
from app.agent.research_agent import run as run_research
from app.agent.graph import _extract_tokens
from app.config.settings import settings
from app.observability.costlogger import log_generation


logger = logging.getLogger(__name__)
_MODEL_NAME = "llama-3.3-70b-versatile"
langfuse = get_client()
llm = ChatGroq(model=_MODEL_NAME, api_key=settings.GROQ_API_KEY, temperature=0)

_SYSTEM_PROMPT = """
You are an orchestrator and planning agent for company reasearch tool. 

You need to follow these. 
1. You are a planning agent. 
2. You are given a question asks some information about a company, break the question in to 2-3 sub-questions. 
3. Assign one of these types to each: web_research, github, general
4. Do not answer the questions — only produce the plan
5. Return only valid JSON in this format:

{
  "sub_questions": [
    {"question": "Question text", "type": "web_research"}
  ]
}
"""

def _parse_sub_questions(raw: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"```json\s*|\s*```","",raw.strip()).strip()
    parsed = json.loads(cleaned)
    sub_questions = parsed.get('sub_questions',[])

    if not isinstance(sub_questions, list) or not sub_questions:
        raise ValueError("No sub-questions returned.")

    valid_types = ['web_research', 'github', 'general']
    cleaned_questions = []

    for i,sub_question in enumerate(sub_questions):
        current_ques:Dict[str,Any]= {}

        if sub_question.get('question'):
            current_ques['question'] = sub_question.get('question')
        else:
            continue

        if sub_question.get('type') not in valid_types:
            current_ques['type'] = 'general'
        else:
            current_ques['type'] = sub_question.get('type')

        current_ques['id'] = i+1

        cleaned_questions.append(current_ques)

    return cleaned_questions

async def _dispatch( sub_question: dict[str, Any],company: str,session_id: str,github_repo: str | None)->dict:
    question = sub_question.get('question')
    if sub_question.get('type')=='github':
        if github_repo:
            return await run_github(question=question,company=company,session_id=session_id)

    """ 
    we are getting this data 
    {
        'answer':final_state.get('answer') or 'no answer produced',
        'steps':final_state.get('steps',0),
        'observations':final_state.get('observations',[]),
        "cost_usd": final_state.get("cost_usd", 0.0),
        "sources_used": _extract_sources(final_state.get("observations", [])),
        "tool_calls_log": final_state.get("messages", []),
    }
    """
    return await run_research(question,company,session_id)

def _merge( original_question: str, sub_questions: list[dict[str, Any]], results: list[dict], ) -> dict:
    sections = [f"Research results for: {original_question}\n"]
    all_sources: list[str] = []
    total_cost: float = 0.0
    total_steps: int = 0

    for sub_q, result in zip(sub_questions, results):
        sections.append(f"[{sub_q['type'].upper()}] {sub_q['question']}")
        sections.append(result.get("answer", ""))
        sections.append("")

        all_sources.extend(result.get("sources_used", []))
        total_cost += result.get("cost_usd", 0.0)
        total_steps += result.get("steps", 0)

    return {
        "answer": "\n".join(sections).strip(),
        "sources_used": list(dict.fromkeys(all_sources)),
        "tool_calls_log": [],
        "steps": total_steps,
        "cost_usd": total_cost
    }


@observe(name="orchestrator")
async def run_orchestrator(
    question: str,
    company: str,
    session_id: str,
    github_repo: str | None = None,
) -> dict:
    with propagate_attributes(session_id=session_id,metadata={"company": company, "orchestrator": True}):
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Company: {company}\nUser question: {question}"}

        ]
        response = await llm.ainvoke(messages)

        raw = response.content.strip()

        input_tokens, output_tokens = _extract_tokens(response)
        plan_cost = log_generation(model=_MODEL_NAME, input_tokens=input_tokens, output_tokens=output_tokens )
        langfuse.update_current_generation(
                model=_MODEL_NAME,
                usage_details={"input_tokens": input_tokens, "output_tokens": output_tokens},
                metadata={"phase": "planning", "plan_cost_usd": plan_cost}
        )

        sub_questions = _parse_sub_questions(raw)
        tasks = [
            _dispatch(sq, company, session_id, github_repo)
            for sq in sub_questions
        ]
        results: list[dict] = await asyncio.gather(*tasks)

        merged = _merge(question, sub_questions, results)
        merged["cost_usd"] += plan_cost

        return merged