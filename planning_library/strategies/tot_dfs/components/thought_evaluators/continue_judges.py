from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import AsyncCallbackManager, CallbackManager


class BaseThoughtEvaluatorContinueJudge(ABC):
    """A base thought evaluator continuation judge. It is responsible for determining based on the value
    if the thought should be explored further or discarded."""

    @abstractmethod
    def should_continue(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        next_thought: List[AgentAction] | AgentAction | AgentFinish,
        value: Any,
        run_manager: Optional[CallbackManager],
    ) -> bool: ...

    @abstractmethod
    async def ashould_continue(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        next_thought: List[AgentAction] | AgentAction | AgentFinish,
        value: Any,
        run_manager: Optional[AsyncCallbackManager],
    ) -> bool: ...


class ThresholdThoughtEvaluatorContinueJudge(BaseThoughtEvaluatorContinueJudge):
    """A thought evaluator continuation judge powered by simple thresholding logic: continue when value is high enough.

    Expects the corresponding backbone to return floats."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def should_continue(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        next_thought: List[AgentAction] | AgentAction | AgentFinish,
        value: float,
        run_manager: Optional[CallbackManager],
    ) -> bool:
        return value > self.threshold

    async def ashould_continue(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        next_thought: List[AgentAction] | AgentAction | AgentFinish,
        value: float,
        run_manager: Optional[AsyncCallbackManager],
    ) -> bool:
        return value > self.threshold
