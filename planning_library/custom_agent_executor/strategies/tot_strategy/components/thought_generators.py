import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager


class BaseThoughtGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[Union[AgentAction, List[AgentAction], AgentFinish]]:
        ...

    @abstractmethod
    async def agenerate(
        self,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[Union[AgentAction, List[AgentAction], AgentFinish]]:
        ...


class AgentThoughtGenerator(BaseThoughtGenerator):
    def generate(
        self,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[Union[AgentAction, List[AgentAction], AgentFinish]]:
        results = [
            agent.plan(
                intermediate_steps=trajectory,
                callbacks=run_manager,
                **inputs,
            )
            for _ in range(max_num_thoughts)
        ]
        return results

    async def agenerate(
        self,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[Union[AgentAction, List[AgentAction], AgentFinish]]:
        # TODO: no idea why mypy complains
        with asyncio.TaskGroup() as tg:  # type: ignore[attr-defined]
            tasks = [
                tg.create_task(
                    agent.aplan(
                        intermediate_steps=trajectory,
                        callbacks=run_manager,
                        **inputs,
                    )
                )
                for _ in range(max_num_thoughts)
            ]

        return [task.result() for task in tasks]
