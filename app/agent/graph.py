import json
import re
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from app.agent.state import AgentState
from app.agent.tools import query_rag_tool, search_web_tool, get_github_tool
from app.config.settings import settings
from langgraph.graph import START

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

SYSTEM_PROMPT = """You are a company research assistant. Your job is to research companies and answer questions about them.

You have access to these tools:
- search_web: Search the web for company info and store in RAG
- get_github: Fetch GitHub activity for a company's repo
- query_rag: Query already-ingested company data

Strategy:
1. Always start by calling query_rag if data is less relevant then call search_web to gather fresh data then call query_rag to retrieve relevant chunks and form your answer
2. Use get_github only if the question is about code, releases, or commits
3. After gathering enough data, return your final answer

Respond ONLY with valid JSON in one of these two formats:

To call a tool:
{"tool": "<tool_name>", "tool_input": {<args>}}

To give the final answer:
{"answer": "<your final answer here>"}
"""


def _parse_json(data: str):
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", data)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            raise Exception("invalid json data provided")

    raise Exception("invalid json data provided")


async def reason(state: AgentState):
    print("hello from reason\n")
    messages = [{'role': 'system', "content": SYSTEM_PROMPT}]

    history_messages = state.get('messages', [])

    if history_messages:
        messages.extend(history_messages)
    else:
        session_id, company = state.get('company', "::").split("::")
        user_message = (f"Research this company: {company}\n\n"
                        f"Question: {state.get('question')}\n\n"
                        f"Session ID: {session_id}")
        messages.append({"role": "user", "content": user_message})

    for obs in state.get('observations', []):
        messages.append({'role': "user", "content": f'Tool result:{obs}'})

    response = await llm.ainvoke(messages)

    print(response,end="\n__________\n")
    raw = response.content.strip()

    state['messages'].append({"role": "assistant", "content": raw})

    parsed_res = _parse_json(raw)

    if 'answer' in parsed_res:
        state['answer'] = parsed_res['answer']
    elif "tool" in parsed_res:
        state['tool_calls'] = [parsed_res]

    return {
        **state,
        "steps": state.get("steps", 0) + 1,
    }


async def act(state: AgentState) -> AgentState:
    tool_calls = list(state.get("tool_calls", []))
    if not tool_calls:
        return state

    call = tool_calls.pop(0)
    tool_name = call["tool"]
    tool_input = call.get("tool_input", {})

    parts = state["company"].split("::", 1)
    session_id = parts[0]
    company = parts[1] if len(parts) > 1 else parts[0]

    print(f"act: calling tool '{tool_name}' with input {tool_input}\n")

    if tool_name == "query_rag":
        result = await query_rag_tool(
            query=tool_input.get("query", state["question"]),
            company=company,
            session_id=session_id,
        )

    elif tool_name == "search_web":
        result = await search_web_tool(
            query=tool_input.get("query", state["question"]),
            company=company,
            session_id=session_id,
        )

    elif tool_name == "get_github":
        result = await get_github_tool(
            repo=tool_input["repo"],
            since_days=tool_input.get("since_days", 7),
            company=company,
            session_id=session_id,
        )

    else:
        result = f"Unknown tool requested: {tool_name}"

    return {
        **state,
        "tool_calls": tool_calls,
        "_pending_observation": result,
    }

def observe(state: AgentState) -> AgentState:
    observations = list(state.get("observations", []))
    pending = state.get("_pending_observation")

    if pending is not None:
        if hasattr(pending, "model_dump"):
            observations.append(json.dumps(pending.model_dump(), default=str))
        else:
            observations.append(str(pending))

    return {
        **state,
        "observations": observations,
        "_pending_observation": None,
    }

def decide(state: AgentState) -> str:
    if state["steps"] >= 5:
        return "end"
    if not state["tool_calls"]:
        return "end"
    return "act"


graph = StateGraph(AgentState)
graph.add_node("reason", reason)
graph.add_node("act", act)
graph.add_node("observe", observe)

graph.add_edge(START, "reason")

graph.add_conditional_edges("reason", decide, {"act": "act", "end": END})
graph.add_edge("act", "observe")
graph.add_edge("observe", "reason")

agent = graph.compile()


def _company_key(session_id: str, company: str) -> str:
    return f"{session_id}::{company.lower().strip()}"



async def run_agent(question: str, company: str, session_id: str) -> dict:
    company_key = _company_key(session_id, company)
    initial_state: AgentState = {
        "question": question,
        "company": company_key,
        "messages": [],
        "tool_calls": [],
        "observations": [],
        "answer": '',
        "steps": 0,
        "cost_usd":0.0
    }

    final_state = await agent.ainvoke(initial_state)

    return {
        "answer": final_state.get("final_answer") or "No answer produced.",
        "steps": final_state.get("steps", 0),
        "observations": final_state.get("observations", []),
        "cost_usd": final_state.get("cost_usd", 0.0),
    }

