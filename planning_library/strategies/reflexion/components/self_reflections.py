from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from langchain_core.agents import AgentAction
from langchain_core.runnables import Runnable


class BaseSelfReflection(ABC):
    @abstractmethod
    def self_reflect(
        self, inputs: Dict[str, Any], intermediate_steps: List[Tuple[AgentAction, str]], evaluator_score: Any
    ) -> str:
        ...

    @abstractmethod
    async def aself_reflect(
        self, inputs: Dict[str, Any], intermediate_steps: List[Tuple[AgentAction, str]], evaluator_score: Any
    ) -> str:
        pass


class RunnableSelfReflection(BaseSelfReflection):
    def __init__(self, llm_chain: Runnable[Dict[str, Any], str]):
        self.llm_chain = llm_chain

    def self_reflect(
        self, inputs: Dict[str, Any], intermediate_steps: List[Tuple[AgentAction, str]], evaluator_score: Any
    ) -> str:
        return self.llm_chain.invoke(
            {
                "inputs": inputs,
                "intermediate_steps": intermediate_steps,
                "evaluator_score": evaluator_score,
            }
        )

    async def aself_reflect(
        self, inputs: Dict[str, Any], intermediate_steps: List[Tuple[AgentAction, str]], evaluator_score: Any
    ) -> str:
        return await self.llm_chain.ainvoke(
            {
                "inputs": inputs,
                "intermediate_steps": intermediate_steps,
                "evaluator_score": evaluator_score,
            }
        )
