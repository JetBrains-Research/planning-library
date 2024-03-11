import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple, Union

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_executor import ToolExecutor  # type: ignore[import]

from planning_library.utils import aexecute_tools, execute_tools


class BaseActor(ABC):
    @abstractmethod
    def act(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        self_reflections: Sequence[str],
        **kwargs,
    ) -> Union[List[AgentAction], AgentAction, AgentFinish]:
        ...

    @abstractmethod
    async def aact(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        self_reflections: Sequence[str],
        **kwargs,
    ) -> Union[List[AgentAction], AgentAction, AgentFinish]:
        ...

    @abstractmethod
    def execute_tools(
        self, action: Union[List[AgentAction], AgentAction], **kwargs
    ) -> Union[List[Tuple[AgentAction, str]], Tuple[AgentAction, str]]:
        ...

    @abstractmethod
    async def aexecute_tools(
        self, action: Union[List[AgentAction], AgentAction], **kwargs
    ) -> Union[List[Tuple[AgentAction, str]], Tuple[AgentAction, str]]:
        ...


class AgentActor(BaseActor):
    def __init__(self, agent: Union[BaseSingleActionAgent, BaseMultiActionAgent], tools: Sequence[BaseTool]):
        self.agent = agent
        self.tool_executor = ToolExecutor(tools)

    def act(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        self_reflections: Sequence[str],
        **kwargs,
    ) -> Union[List[AgentAction], AgentAction, AgentFinish]:
        return self.agent.plan(intermediate_steps=intermediate_steps, **inputs, self_reflections=self_reflections)

    def execute_tools(
        self, action: Union[List[AgentAction], AgentAction], **kwargs
    ) -> Union[List[Tuple[AgentAction, str]], Tuple[AgentAction, str]]:
        if isinstance(action, list):
            return [execute_tools(single_action, self.tool_executor) for single_action in action]

        return execute_tools(action, self.tool_executor)

    async def aact(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        self_reflections: Sequence[str],
        **kwargs,
    ) -> Union[List[AgentAction], AgentAction, AgentFinish]:
        return await self.agent.aplan(
            intermediate_steps=intermediate_steps, **inputs, self_reflections=self_reflections
        )

    async def aexecute_tools(
        self, action: Union[List[AgentAction], AgentAction], **kwargs
    ) -> Union[List[Tuple[AgentAction, str]], Tuple[AgentAction, str]]:
        if isinstance(action, list):
            with asyncio.TaskGroup() as tg:  # type: ignore[attr-defined]
                tasks = [tg.create_task(self.aexecute_tools(single_action)) for single_action in action]
            return [task.result() for task in tasks]

        return await aexecute_tools(action, self.tool_executor)
