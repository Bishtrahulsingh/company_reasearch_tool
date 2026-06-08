from app.agent.graph import build_agent_graph, run_agent
from app.agent.tools import search_web_tool, query_rag_tool



SYSTEM_PROMPT = """You are a specialist web research agent. Your job is to find accurate,
sourced information about a company using web search and a RAG knowledge base.

You have access to these tools:
- search_web : search the web for fresh company information and store in RAG
- query_rag  : query already-ingested company data from the knowledge base

Strategy:
1. Always call query_rag first — check what is already known before fetching fresh data
2. If query_rag returns thin or outdated results, call search_web then query_rag again
3. Prefer multiple distinct sources — never rely on a single URL
4. Always include source URLs in your answer

Respond ONLY with valid JSON in one of these two formats:

To call a tool:
{"tool": "<tool_name>", "tool_input": {<args>}}

To give the final answer:
{"answer": "<your answer with sources cited>"}
"""

graph = build_agent_graph(
    system_prompt=SYSTEM_PROMPT,
    tools={
        "search_web": search_web_tool,
        "query_rag": query_rag_tool,
    },
    max_steps=3,
)


async def run(question: str, company: str, session_id: str) -> dict:
    return await run_agent(graph, question, company, session_id)