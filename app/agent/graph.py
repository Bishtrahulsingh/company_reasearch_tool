import asyncio
import json
import logging
import re
import uuid
from json import JSONDecodeError
from typing import Union, Tuple, Any, List, Dict

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import get_client, observe, propagate_attributes
from langgraph.graph import END, START, StateGraph

from app.agent.sanitizer import sanitize_tool_result
from app.agent.state import AgentState
from app.config.settings import settings
from app.observability.costlogger import log_generation


logger = logging.getLogger(__name__)
_TOOL_TIMEOUT = 20.0

if settings.LLM_PROVIDER == 'groq':
    _MODEL_NAME = 'llama-3.3-70b-versatile'
else:
    _MODEL_NAME = 'gemini-2.0-flash'

_LLM_SEMAPHORE = asyncio.Semaphore(2)

langfuse = get_client()


def _parse_json(data: str):
    if not data or not data.strip():
        raise Exception('LLM returned an empty response')

    cleaned = re.sub(r'```json\s*|\s*```', '', data.strip()).strip()

    if not cleaned:
        raise Exception('LLM returned only markdown fences with no content')

    try:
        return json.loads(cleaned)
    except JSONDecodeError as e:
        decoder = json.JSONDecoder()
        for start in range(len(cleaned)):
            if cleaned[start] in '{[':
                try:
                    obj, _ = decoder.raw_decode(cleaned, start)
                    return obj
                except JSONDecodeError:
                    continue
        raise Exception(f'invalid json data found. Raw (first 300 chars): {cleaned[:300]!r}') from e


def _company_key(session_id: Union[str, uuid.UUID], company: str) -> str:
    return f'{session_id}::{company}'


def _extract_tokens(response) -> Tuple[int, int]:
    usage = getattr(response, 'usage_metadata', None) or getattr(response, 'response_metadata', {}).get('token_usage', {})

    if not usage:
        return 0, 0

    input_tokens: int = (
        getattr(usage, 'input_tokens', None)
        or getattr(usage, 'prompt_tokens', None)
        or (usage.get('input_tokens') if isinstance(usage, dict) else None)
        or (usage.get('prompt_tokens') if isinstance(usage, dict) else None)
        or 0
    )

    output_tokens: int = (
        getattr(usage, 'output_tokens', None)
        or getattr(usage, 'completion_tokens', None)
        or (usage.get('output_tokens') if isinstance(usage, dict) else None)
        or (usage.get('completion_tokens') if isinstance(usage, dict) else None)
        or 0
    )

    return input_tokens, output_tokens


def _extract_sources(observations: list):
    sources = []
    for o in observations:
        try:
            obs = json.loads(o) if isinstance(o, str) else o
            if isinstance(obs, dict) and obs.get('source'):
                sources.append(obs['source'])
        except Exception:
            pass
    return list(dict.fromkeys(sources))


def build_agent_graph(system_prompt: str, tools: dict, max_steps: int = 5):
    if settings.LLM_PROVIDER == 'groq':
        llm = ChatGroq(
            model=_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
        )
    else:
        llm = ChatGoogleGenerativeAI(
            model=_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0,
        )

    @observe(name='reason', as_type='generation')
    async def reason(state: AgentState):
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        message_history = state.get('messages', [])
        observations = state.get('observations', [])

        if message_history:
            messages.extend(message_history)
        else:
            session_id, company = state.get('company', "::").split("::")
            user_message = (
                f'Research this company: {company}\n'
                f'Question: {state.get("question")}\n'
                f'Session ID: {session_id}\n'
            )
            messages.append({'role': 'user', 'content': user_message})

        for obs in observations:
            messages.append({'role': 'user', 'content': f'Tool result: {obs}'})

        async with _LLM_SEMAPHORE:
            response = await llm.ainvoke(messages)

        raw = response.content.strip()

        input_tokens, output_tokens = _extract_tokens(response)
        step_cost = log_generation(
            model=_MODEL_NAME,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        langfuse.update_current_generation(
            model=_MODEL_NAME,
            usage_details={
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
            },
            metadata={'step_cost_usd': step_cost}
        )

        state['messages'].append({'role': 'assistant', 'content': raw})

        parsed = _parse_json(raw)

        if 'answer' in parsed:
            state['answer'] = parsed['answer']
        elif 'tool' in parsed:
            state['tool_calls'] = [parsed]

        return {
            **state,
            'steps': state.get('steps', 0) + 1,
            'cost_usd': state.get('cost_usd', 0.0) + step_cost,
        }

    @observe(name='act')
    async def act(state: AgentState):
        tool_calls = list(state.get('tool_calls', []))
        if not tool_calls:
            return state

        call = tool_calls.pop(0)
        tool_name = call.get('tool', '')
        tool_input = call.get('tool_input', {})

        session_id, company = state.get('company', '::').split('::')

        tool_input.setdefault('session_id', session_id)
        tool_input.setdefault('company', company)

        tool_fn = tools.get(tool_name)

        if tool_fn is None:
            result = f'unknown tool: {tool_name}'
        else:
            try:
                result = await asyncio.wait_for(
                    tool_fn(**tool_input),
                    timeout=_TOOL_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning("act: tool '%s' timed out", tool_name)
                result = f"Tool '{tool_name}' timed out, skipping."

        return {
            **state,
            'tool_calls': tool_calls,
            '_pending_observation': result,
        }

    @observe(name='observe')
    def observe_node(state: AgentState):
        observations = list(state.get('observations', []))
        pending = state.get('_pending_observation', None)

        if pending is not None:
            if hasattr(pending, 'model_dump'):
                raw = json.dumps(pending.model_dump(), default=str)
            else:
                raw = str(pending)
            observations.append(sanitize_tool_result(raw))

        return {
            **state,
            'observations': observations,
            '_pending_observation': None,
        }

    def decide(state: AgentState):
        if state['steps'] >= max_steps:
            return 'end'
        if not state.get('tool_calls'):
            return 'end'
        return 'act'

    graph = StateGraph(AgentState)

    graph.add_node('reason', reason)
    graph.add_node('act', act)
    graph.add_node('observe', observe_node)

    graph.add_edge(START, 'reason')
    graph.add_edge('observe', 'reason')
    graph.add_edge('act', 'observe')

    graph.add_conditional_edges('reason', decide, {'act': 'act', 'end': END})

    return graph.compile()


@observe(name='run_agent')
async def run_agent(compiled_graph, question: str, company: str, session_id: str) -> dict:
    with propagate_attributes(session_id=session_id, metadata={'company': company}):
        company_key = _company_key(session_id, company)
        initial_state: AgentState = {
            'question': question,
            'company': company_key,
            'messages': [],
            'tool_calls': [],
            'observations': [],
            'answer': '',
            'steps': 0,
            'cost_usd': 0.0,
            '_pending_observation': None,
        }

        final_state = await compiled_graph.ainvoke(initial_state)

        return {
            'answer': final_state.get('answer') or 'no answer produced',
            'steps': final_state.get('steps', 0),
            'observations': final_state.get('observations', []),
            'cost_usd': final_state.get('cost_usd', 0.0),
            'sources_used': _extract_sources(final_state.get('observations', [])),
            'tool_calls_log': final_state.get('messages', []),
        }