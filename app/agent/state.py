from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    question:str
    messages:List[Dict[str,str]]
    tool_calls:List[Dict[str,Any]]
    observations:List[str]
    answer:str
    steps:int
    cost_usd:int