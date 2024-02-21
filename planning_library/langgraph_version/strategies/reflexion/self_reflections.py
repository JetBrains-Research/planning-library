from abc import ABC, abstractmethod
from typing import Any, Dict

from langchain_core.runnables import Runnable

from planning_library.langgraph_version.utils import AgentState


class BaseSelfReflection(ABC):
    @abstractmethod
    def self_reflect(self, state: AgentState, **kwargs) -> AgentState:
        pass

    @abstractmethod
    async def aself_reflect(self, state: AgentState, **kwargs) -> AgentState:
        pass


class RunnableSelfReflection(BaseSelfReflection):
    def __init__(self, llm_chain: Runnable[Dict[str, Any], str]):
        self.llm_chain = llm_chain

    def self_reflect(self, state: AgentState, **kwargs) -> AgentState:
        llm_output = self.llm_chain.invoke(
            {
                "inputs": state["inputs"],
                "evaluator_score": state["evaluator_score"],
                "agent_outcome": state["agent_outcome"],
                "intermediate_steps": state["intermediate_steps"],
            }
        )

        return {
            "inputs": state["inputs"],
            "agent_outcome": state["agent_outcome"],
            "evaluator_score": state["evaluator_score"],
            "self_reflections": [llm_output],
            "intermediate_steps": [],
            "iteration": state["iteration"],
        }

    async def aself_reflect(self, state: AgentState, **kwargs) -> AgentState:
        llm_output = await self.llm_chain.ainvoke(
            {
                "inputs": state["inputs"],
                "evaluator_score": state["evaluator_score"],
                "agent_outcome": state["agent_outcome"],
                "intermediate_steps": state["intermediate_steps"],
            }
        )
        return {
            "inputs": state["inputs"],
            "agent_outcome": state["agent_outcome"],
            "evaluator_score": state["evaluator_score"],
            "self_reflections": [llm_output],
            "intermediate_steps": [],
            "iteration": state["iteration"],
        }
