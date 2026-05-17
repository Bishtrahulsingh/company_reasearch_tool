from langgraph.graph import StateGraph, END
from app.agent.state import AgentState

def reason(state: AgentState) -> AgentState:
    pass

def act(state: AgentState) -> AgentState:
    pass

def observe(state: AgentState) -> AgentState:
    pass

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

graph.set_entry_point("reason")
graph.add_conditional_edges("reason", decide, {"act": "act", "end": END})
graph.add_edge("act", "observe")
graph.add_edge("observe", "reason")

agent = graph.compile()