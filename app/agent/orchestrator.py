import json
import re
from typing import Any
from langchain_groq import ChatGroq
from app.agent.base_agent import AgentResult, BaseAgent
from app.config.settings import settings
from app.observability.costlogger import log_generation


MODEL_NAME = "llama-3.3-70b-versatile"

llm = ChatGroq(
    model=MODEL_NAME,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)


SYSTEM_PROMPT = """
You are the planning agent of a company research system.

Break the user's company-related question into 2-3 small, independent
sub-questions.

Assign one type to each question:

- web_research: recent news, funding, leadership, products, or live information
- github: commits, releases, issues, or repository activity
- general: information that can be answered from existing RAG documents

Do not answer the questions.

Return only valid JSON in this format:

{
  "sub_questions": [
    {
      "question": "Question text",
      "type": "web_research"
    }
  ]
}
"""


def parse_sub_questions(raw_response: str) -> list[dict[str, Any]]:
    cleaned_response = re.sub(
        r"```(?:json)?\s*|\s*```",
        "",
        raw_response,
    ).strip()

    parsed_response = json.loads(cleaned_response)
    sub_questions = parsed_response.get("sub_questions", [])

    if not isinstance(sub_questions, list) or not sub_questions:
        raise ValueError("No sub-questions found.")

    valid_types = {"web_research", "github", "general"}
    validated_questions = []

    for index, item in enumerate(sub_questions[:3], start=1):
        question = item.get("question", "").strip()
        question_type = item.get("type", "general")

        if not question:
            raise ValueError(f"Invalid sub-question: {item}")

        if question_type not in valid_types:
            question_type = "general"

        validated_questions.append(
            {
                "id": index,
                "question": question,
                "type": question_type,
            }
        )

    return validated_questions


class OrchestratorAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="OrchestratorAgent")

    async def _run(self,question: str,context: dict[str, Any]) -> AgentResult:
        company = context.get("company", "the company")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Company: {company}\nUser question: {question}",
            },
        ]

        response = await llm.ainvoke(messages)
        raw_response = response.content.strip()

        sub_questions = parse_sub_questions(raw_response)

        context["sub_questions"] = sub_questions

        usage = response.usage_metadata or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        cost_usd = log_generation(
            model=MODEL_NAME,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        plan_lines = [f'Research plan for: "{question}"']

        for item in sub_questions:
            plan_lines.append(
                f"[{item['type'].upper()}] "
                f"{item['id']}. {item['question']}"
            )

        return AgentResult(
            answer="\n".join(plan_lines),
            sources=[],
            cost_usd=cost_usd,
            agent_name=self.name,
            metadata={
                "sub_questions": sub_questions,
            },
        )