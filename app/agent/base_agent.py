import abc
import time
from dataclasses import Field
from typing import List, Any

import logging
from pydantic import BaseModel


logger = logging.getLogger(__name__)

class AgentResult(BaseModel):
    answer:str
    sources:List[str]
    cost_usd:float=0.0
    agent_name:str
    metadata:dict[str, Any]
    latency_ms:float=0.0


class BaseAgent(abc.ABC):
    def __init__(self,name:str):
        self.name = name
        self._logger = logging.getlogger(f'agent:{name}')

    @abc.abstractmethod
    async def _run(self, question: str, context: dict[str, Any]) -> AgentResult:
        pass

    async def run(self, question:str, context:dict[str,Any])->AgentResult:
        self._logger.info(f'starting : question={question}')
        t0 = time.perf_counter()

        try:
            result = await self._run(question, context)
        except Exception as e:
            self._logger.exception(f'unhandled exception: {e}')
            result = AgentResult(
                answer=f'failed with error: {e} ',
                agent_name= self.name,
                cost_usd= 0,
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        result.agent_name = self.name
        result.latency_ms = round(latency_ms, 2)

        self._logger.info(
            "Done | latency_ms=%.0f cost_usd=%.6f sources=%d",
            result.latency_ms,
            result.cost_usd,
            len(result.sources),
        )

        return result


