from abc import ABC, abstractmethod
from langchain_core.callbacks import (
    CallbackManager,
    AsyncCallbackManager,
)
from typing import Optional, Tuple, List, Dict, Any, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from planning_library.strategies import BaseCustomStrategy


class BaseADaPTExecutor(ABC):
    @abstractmethod
    @property
    def agent(self) -> Union[BaseSingleActionAgent, BaseMultiActionAgent]: ...

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

    @property
    def agent(self) -> Union[BaseSingleActionAgent, BaseMultiActionAgent]:
        return self._executor.agent

    def _is_completed(self, outcome: AgentFinish) -> bool:
        return "task completed" in outcome.log.lower()

    def _process_strategy_outputs(
        self, outputs: Dict[str, Any]
    ) -> Tuple[bool, AgentFinish, List[Tuple[AgentAction, str]]]:
        outcome = AgentFinish(
            return_values={
                key: value[0]
                for key, value in outputs.items()
                if isinstance(key, list)
                and key not in ["finish_log", "intermediate_steps"]
            },
            log=outputs["finish_log"][0],
        )
        is_completed = self._is_completed(outcome)
        return is_completed, outcome, outputs["intermediate_steps"][0]

    def execute(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManager] = None,
    ) -> Tuple[bool, AgentFinish, List[Tuple[AgentAction, str]]]:
        outputs = self._executor.invoke(
            inputs,
            config={"callbacks": run_manager} if run_manager else {},
        )
        return self._process_strategy_outputs(outputs)

    async def aexecute(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> Tuple[bool, AgentFinish, List[Tuple[AgentAction, str]]]:
        outputs = await self._executor.ainvoke(
            inputs,
            config={"callbacks": run_manager} if run_manager else {},
        )
        return self._process_strategy_outputs(outputs)
