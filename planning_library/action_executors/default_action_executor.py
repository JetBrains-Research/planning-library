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
from .meta_tools import MetaTools


class LangchainActionExecutor(BaseActionExecutor):
    def __init__(
        self, tools: Sequence[BaseTool], meta_tools: Optional[MetaTools] = None
    ):
        self._tool_executor = ToolExecutor(tools)
        self._meta_tool_executor = (
            ToolExecutor(meta_tools.tools) if meta_tools else None
        )
        self._meta_tool_names = meta_tools.tool_names_map if meta_tools else {}

    @property
    def tools(self) -> Sequence[BaseTool]:
        return self._tool_executor.tools

    @property
    def reset_tool_name(self) -> Optional[str]:
        if "reset" in self._meta_tool_names:
            return self._meta_tool_names["reset"]
        return None

    def reset(
        self,
        actions: Optional[List[AgentAction]] = None,
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> None:
        """Resets the current state. If actions are passed, will also execute them."""
        if self.reset_tool_name is not None:
            self.execute(
                actions=[
                    AgentAction(
                        tool=self.reset_tool_name,
                        tool_input={},
                        log="Invoking reset tool.",
                    )
                ],
                run_manager=run_manager,
            )
            if actions:
                self.execute(actions, run_manager=run_manager)

    @overload
    def execute(
        self,
        actions: List[AgentAction],
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    def execute(
        self,
        actions: AgentAction,
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> AgentStep: ...

    def execute(
        self,
        actions: List[AgentAction] | AgentAction,
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> List[AgentStep] | AgentStep:
        if isinstance(actions, list):
            observations = [
                self.execute(action, run_manager=run_manager) for action in actions
            ]
        else:
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

    async def areset(
        self,
        actions: Optional[List[AgentAction]] = None,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> None:
        """Resets the current state. If actions are passed, will also execute them."""
        if self.reset_tool_name is not None:
            await self.aexecute(
                actions=[
                    AgentAction(
                        tool=self.reset_tool_name,
                        tool_input={},
                        log="Invoking reset tool.",
                    )
                ],
                run_manager=run_manager,
            )
            if actions:
                await self.aexecute(actions, run_manager=run_manager)

    @overload
    async def aexecute(
        self,
        actions: List[AgentAction],
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    async def aexecute(
        self,
        actions: AgentAction,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> AgentStep: ...

    async def aexecute(
        self,
        actions: List[AgentAction] | AgentAction,
        run_manager: Optional[AsyncCallbackManager] = None,
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
