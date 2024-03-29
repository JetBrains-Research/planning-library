from abc import ABC, abstractmethod
from langchain_core.callbacks import (
    CallbackManager,
    AsyncCallbackManager,
)
from typing import Optional, Tuple, List, Dict, Any
from langchain_core.agents import AgentAction, AgentFinish
from planning_library.strategies.adapt.utils import ADaPTPlan, InitialADaPTPlan
from langchain_core.runnables import Runnable


class BaseADaPTPlanner(ABC):
    @abstractmethod
    def plan(
        self,
        inputs: Dict[str, Any],
        current_depth: int,
        agent_outcome: AgentFinish,
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManager] = None,
    ) -> ADaPTPlan: ...

    @abstractmethod
    async def aplan(
        self,
        inputs: Dict[str, Any],
        current_depth: int,
        agent_outcome: AgentFinish,
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> ADaPTPlan: ...


class RunnableADaPTPlanner(BaseADaPTPlanner):
    def __init__(self, runnable: Runnable[Dict[str, Any], InitialADaPTPlan]):
        self.runnable = runnable

    def plan(
        self,
        inputs: Dict[str, Any],
        current_depth: int,
        agent_outcome: AgentFinish,
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManager] = None,
    ) -> ADaPTPlan:
        initial_plan = self.runnable.invoke(
            {
                **inputs,
                "agent_outcome": agent_outcome,
                "intermediate_steps": intermediate_steps,
            },
            {"callbacks": run_manager} if run_manager else {},
        )
        return {
            "subtasks": [
                {"inputs": task, "depth": current_depth + 1}
                for task in initial_plan["subtasks"]
            ],
            "logic": initial_plan["logic"],
        }

    async def aplan(
        self,
        inputs: Dict[str, Any],
        current_depth: int,
        agent_outcome: AgentFinish,
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> ADaPTPlan:
        initial_plan = await self.runnable.ainvoke(
            {
                **inputs,
                "agent_outcome": agent_outcome,
                "intermediate_steps": intermediate_steps,
            },
            {"callbacks": run_manager} if run_manager else {},
        )
        return {
            "subtasks": [
                {"inputs": task, "depth": current_depth + 1}
                for task in initial_plan["subtasks"]
            ],
            "logic": initial_plan["logic"],
        }
