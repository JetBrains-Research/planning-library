import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple, Union

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager


class BaseThoughtGenerator(ABC):
    def __init__(self, generation_mode: Literal["sample", "propose"] = "sample"):
        self.generation_mode = generation_mode

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
        # sample: `max_num_thoughts` i.i.d. requests
        if self.generation_mode == "sample":
            results: List[Union[AgentAction, List[AgentAction], AgentFinish]] = []
            for _ in range(max_num_thoughts):
                cur_result = agent.plan(
                    intermediate_steps=trajectory,
                    callbacks=run_manager,
                    previous_thoughts=results,
                    **inputs,
                )
                results.append(cur_result)

            return results

        # TODO propose: a single request that should return `max_num_thoughts` thoughts
        raise ValueError("Current thought generation mode is not supported yet.")

    async def agenerate(
        self,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[Union[AgentAction, List[AgentAction], AgentFinish]]:
        # sample: `max_num_thoughts` i.i.d. requests
        if self.generation_mode == "sample":
            # TODO: don't repeat suggestions in async version
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

        # TODO propose: a single request that should return `max_num_thoughts` thoughts
        raise ValueError("Current thought generation mode is not supported yet.")
