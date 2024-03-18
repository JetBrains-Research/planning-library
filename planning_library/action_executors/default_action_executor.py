from typing import List, overload, Sequence

from langchain_core.agents import AgentAction, AgentStep
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_executor import ToolExecutor  # type: ignore[import-untyped]
from .base_action_executor import BaseActionExecutor


class DefaultActionExecutor(BaseActionExecutor):
    def __init__(self, tools: Sequence[BaseTool]):
        self._tool_executor = ToolExecutor(tools)

    @property
    def tools(self) -> Sequence[BaseTool]:
        return self._tool_executor.tools

    @overload
    def execute(
        self,
        actions: List[AgentAction],
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    def execute(
        self,
        actions: AgentAction,
        **kwargs,
    ) -> AgentStep: ...

    def execute(
        self,
        actions: List[AgentAction] | AgentAction,
        **kwargs,
    ) -> List[AgentStep] | AgentStep:
        observations = self._tool_executor.invoke(actions)
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
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    async def aexecute(
        self,
        actions: AgentAction,
        **kwargs,
    ) -> AgentStep: ...

    async def aexecute(
        self,
        actions: List[AgentAction] | AgentAction,
        **kwargs,
    ) -> List[AgentStep] | AgentStep:
        observations = await self._tool_executor.ainvoke(actions)
        if isinstance(observations, list):
            assert isinstance(actions, list)
            return [
                AgentStep(action=action, observation=observation)
                for action, observation in zip(actions, observations)
            ]
        assert isinstance(actions, AgentAction)
        return AgentStep(action=actions, observation=observations)
