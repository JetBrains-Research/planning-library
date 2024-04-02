from __future__ import annotations
from typing import List, overload, Sequence, Optional

from langchain_core.agents import AgentAction, AgentStep
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_executor import ToolExecutor  # type: ignore[import-untyped]
from .base_action_executor import BaseActionExecutor
from langchain_core.callbacks import (
    CallbackManager,
    AsyncCallbackManager,
)


class DefaultActionExecutor(BaseActionExecutor):
    def __init__(self, tools: Sequence[BaseTool]):
        self._tool_executor = ToolExecutor(tools)

    def reset(self, actions: Optional[List[AgentAction]] = None, **kwargs) -> None:
        """Resets the current state. If actions are passed, will also execute them.

        This action executor doesn't have a state by default, so this method doesn't do anything.
        """
        ...

    @property
    def tools(self) -> Sequence[BaseTool]:
        return self._tool_executor.tools

    @overload
    def execute(
        self,
        actions: List[AgentAction],
        run_manager: Optional[CallbackManager] = None,
        reset_before_action: bool = False,
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    def execute(
        self,
        actions: AgentAction,
        run_manager: Optional[CallbackManager] = None,
        reset_before_action: bool = False,
        **kwargs,
    ) -> AgentStep: ...

    def execute(
        self,
        actions: List[AgentAction] | AgentAction,
        run_manager: Optional[CallbackManager] = None,
        reset_before_action: bool = False,
        **kwargs,
    ) -> List[AgentStep] | AgentStep:
        observations = self._tool_executor.invoke(
            actions,
            config={"callbacks": run_manager} if run_manager else {},
        )
        if isinstance(observations, list):
            assert isinstance(actions, list)
            return [
                AgentStep(action=action, observation=observation)
                for action, observation in zip(actions, observations)
            ]

        assert isinstance(actions, AgentAction)
        return AgentStep(action=actions, observation=observations)

    @overload
    async def aexecute(
        self,
        actions: List[AgentAction],
        run_manager: Optional[AsyncCallbackManager] = None,
        reset_before_action: bool = False,
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    async def aexecute(
        self,
        actions: AgentAction,
        run_manager: Optional[AsyncCallbackManager] = None,
        reset_before_action: bool = False,
        **kwargs,
    ) -> AgentStep: ...

    async def aexecute(
        self,
        actions: List[AgentAction] | AgentAction,
        run_manager: Optional[AsyncCallbackManager] = None,
        reset_before_action: bool = False,
        **kwargs,
    ) -> List[AgentStep] | AgentStep:
        observations = await self._tool_executor.ainvoke(
            actions,
            config={"callbacks": run_manager} if run_manager else {},
        )
        if isinstance(observations, list):
            assert isinstance(actions, list)
            return [
                AgentStep(action=action, observation=observation)
                for action, observation in zip(actions, observations)
            ]
        assert isinstance(actions, AgentAction)
        return AgentStep(action=actions, observation=observations)
