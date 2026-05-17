from app.agent.tools import search_web_tool, get_github_tool, query_rag_tool


TOOLS = {
    "search_web": search_web_tool,
    "get_github": get_github_tool,
    "query_rag":  query_rag_tool,
}

TOOL_SCHEMAS = [
    {
        "name": "search_web",
        "description": "Search the web for information about a company and store results in RAG.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":      {"type": "string",  "description": "Search query"},
                "company":    {"type": "string",  "description": "Company name"},
                "session_id": {"type": "string",  "description": "Current session ID"},
            },
            "required": ["query", "company", "session_id"],
        },
    },
    {
        "name": "get_github",
        "description": "Fetch recent GitHub activity for a repo and store in RAG. Use when the user asks about code, releases or commits.",
        "input_schema": {
            "type": "object",
            "properties": {
                "repo":       {"type": "string",  "description": "GitHub repo in owner/name format e.g. openai/openai-python"},
                "company":    {"type": "string",  "description": "Company the repo belongs to"},
                "session_id": {"type": "string",  "description": "Current session ID"},
                "since_days": {"type": "integer", "description": "How many days back to look (default 7)"},
            },
            "required": ["repo", "company", "session_id"],
        },
    },
    {
        "name": "query_rag",
        "description": "Query already-ingested company data. Always call this after search_web or get_github.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":      {"type": "string", "description": "Question to answer from stored data"},
                "company":    {"type": "string", "description": "Company name"},
                "session_id": {"type": "string", "description": "Current session ID"},
            },
            "required": ["query", "company", "session_id"],
        },
    },
]