from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple, Union

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish


class BaseActor(ABC):
    @abstractmethod
    def act(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        self_reflections: Sequence[str],
        **kwargs,
    ) -> Union[List[AgentAction], AgentAction, AgentFinish]: ...

    @abstractmethod
    async def aact(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        self_reflections: Sequence[str],
        **kwargs,
    ) -> Union[List[AgentAction], AgentAction, AgentFinish]: ...


class AgentActor(BaseActor):
    def __init__(self, agent: Union[BaseSingleActionAgent, BaseMultiActionAgent]):
        self.agent = agent

    def act(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        self_reflections: Sequence[str],
        **kwargs,
    ) -> Union[List[AgentAction], AgentAction, AgentFinish]:
        return self.agent.plan(
            intermediate_steps=intermediate_steps,
            **inputs,
            self_reflections=self_reflections,
        )

    async def aact(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        self_reflections: Sequence[str],
        **kwargs,
    ) -> Union[List[AgentAction], AgentAction, AgentFinish]:
        return await self.agent.aplan(
            intermediate_steps=intermediate_steps,
            **inputs,
            self_reflections=self_reflections,
        )
