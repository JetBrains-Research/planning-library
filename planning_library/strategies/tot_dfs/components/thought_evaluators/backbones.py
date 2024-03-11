from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import AsyncCallbackManager, CallbackManager
from langchain_core.runnables import Runnable

from ...utils import EvaluatorInput


class BaseThoughtEvaluatorBackbone(ABC):
    """A base thought evaluator backbone. It is responsible for actually evaluating each proposed thought."""

    @abstractmethod
    def evaluate(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        next_thought: List[AgentAction] | AgentAction | AgentFinish,
        observation: Optional[List[AgentStep] | AgentStep],
        run_manager: Optional[CallbackManager] = None,
    ) -> Any:
        ...

    @abstractmethod
    async def aevaluate(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        next_thought: List[AgentAction] | AgentAction | AgentFinish,
        observation: Optional[List[AgentStep] | AgentStep],
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> Any:
        ...


class RunnableThoughtEvaluator(BaseThoughtEvaluatorBackbone):
    """A thought evaluator backbone powered by a Runnable."""

    def __init__(self, runnable: Runnable[EvaluatorInput, Any]):
        self.runnable = runnable

    def evaluate(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        next_thought: List[AgentAction] | AgentAction | AgentFinish,
        observation: Optional[List[AgentStep] | AgentStep],
        run_manager: Optional[CallbackManager] = None,
    ) -> Any:
        return self.runnable.invoke(
            {"inputs": inputs, "trajectory": trajectory, "next_thought": next_thought, "observation": observation},
            {"callbacks": run_manager} if run_manager else {},
        )

    async def aevaluate(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        next_thought: List[AgentAction] | AgentAction | AgentFinish,
        observation: Optional[List[AgentStep] | AgentStep],
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> Any:
        result = await self.runnable.ainvoke(
            {"inputs": inputs, "trajectory": trajectory, "next_thought": next_thought, "observation": observation},
            {"callbacks": run_manager} if run_manager else {},
        )
        return result
