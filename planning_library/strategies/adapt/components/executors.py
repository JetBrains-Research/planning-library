from abc import ABC, abstractmethod
from langchain_core.callbacks import (
    CallbackManager,
    AsyncCallbackManager,
)
from typing import Optional, Tuple, List, Dict, Any
from langchain_core.agents import AgentAction, AgentFinish
from planning_library.strategies import BaseCustomStrategy


class BaseADaPTExecutor(ABC):
    @abstractmethod
    def execute(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManager] = None,
    ) -> Tuple[bool, AgentFinish, List[Tuple[AgentAction, str]]]: ...

    @abstractmethod
    async def aexecute(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> Tuple[bool, AgentFinish, List[Tuple[AgentAction, str]]]: ...


class StrategyADaPTExecutor(BaseADaPTExecutor):
    def __init__(self, strategy: BaseCustomStrategy):
        self._executor = strategy

    def _is_completed(self, outcome: AgentFinish) -> bool:
        return "task completed" in outcome.log.lower()

    def execute(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManager] = None,
    ) -> Tuple[bool, AgentFinish, List[Tuple[AgentAction, str]]]:
        outputs = self._executor.invoke(**inputs)["outputs"]
        intermediate_steps = outputs.get("intermediate_steps", [])
        finish_log = outputs.get("finish_log", "")
        del outputs["intermediate_steps"]
        del outputs["finish_log"]
        outcome = AgentFinish(return_values=outputs, log=finish_log)
        is_completed = self._is_completed(outcome)
        return is_completed, outcome, intermediate_steps

    async def aexecute(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> Tuple[bool, AgentFinish, List[Tuple[AgentAction, str]]]:
        outputs = await self._executor.ainvoke(**inputs)
        intermediate_steps = outputs.get("intermediate_steps", [])
        finish_log = outputs.get("finish_log", "")
        del outputs["intermediate_steps"]
        del outputs["finish_log"]
        outcome = AgentFinish(return_values=outputs, log=finish_log)
        is_completed = self._is_completed(outcome)
        return is_completed, outcome, intermediate_steps
