from abc import ABC, abstractmethod
from typing import Any, Dict, Literal

from langchain_core.runnables import Runnable

from planning_library.langgraph_version.utils import AgentState


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, state: AgentState, **kwargs) -> AgentState:
        pass

    @abstractmethod
    async def aevaluate(self, state: AgentState, **kwargs) -> AgentState:
        pass

    @abstractmethod
    def should_continue(self, state: AgentState) -> Literal["yes", "no"]:
        pass


class RunnableEvaluator(BaseEvaluator, ABC):
    def __init__(self, llm_chain: Runnable[Dict[str, Any], Any]):
        self.llm_chain = llm_chain

    def evaluate(self, state: AgentState, **kwargs) -> AgentState:
        llm_output = self.llm_chain.invoke(
            {
                "inputs": state["inputs"],
                "agent_outcome": state["agent_outcome"],
                "intermediate_steps": state["intermediate_steps"],
            },
        )

        return {
            "inputs": state["inputs"],
            "agent_outcome": state["agent_outcome"],
            "evaluator_score": llm_output,
            "self_reflections": [],
            "intermediate_steps": [],
            "iteration": state["iteration"] + 1,
        }

    async def aevaluate(self, state: AgentState, **kwargs) -> AgentState:
        llm_output = await self.llm_chain.ainvoke(
            {
                "inputs": state["inputs"],
                "agent_outcome": state["agent_outcome"],
                "intermediate_steps": state["intermediate_steps"],
            },
        )
        return {
            "inputs": state["inputs"],
            "agent_outcome": state["agent_outcome"],
            "evaluator_score": llm_output,
            "self_reflections": [],
            "intermediate_steps": [],
            "iteration": state["iteration"] + 1,
        }


class BinaryRunnableEvaluator(RunnableEvaluator):
    def __init__(self, llm_chain: Runnable[Dict[str, Any], bool]):
        super().__init__(llm_chain)

    def should_continue(self, state: AgentState) -> Literal["yes", "no"]:
        return state["evaluator_score"]


class ThresholdRunnableEvaluator(RunnableEvaluator):
    def __init__(self, llm_chain: Runnable[Dict[str, Any], float], threshold: float):
        super().__init__(llm_chain)
        self._threshold = threshold

    def should_continue(self, state: AgentState) -> Literal["yes", "no"]:
        return "no" if state["evaluator_score"] >= self._threshold else "yes"
