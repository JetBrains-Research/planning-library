import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Tuple, overload

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import AsyncCallbackManager, CallbackManager


class BaseThoughtGenerator(ABC):
    def __init__(self, generation_mode: Literal["sample", "propose"] = "sample"):
        self.generation_mode = generation_mode

    @overload
    def generate(
        self,
        agent: BaseSingleActionAgent,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[AgentAction | AgentFinish]:
        ...

    @overload
    def generate(
        self,
        agent: BaseMultiActionAgent,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[List[AgentAction] | AgentFinish]:
        ...

    @abstractmethod
    def generate(
        self,
        agent: BaseSingleActionAgent | BaseMultiActionAgent,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[List[AgentAction] | AgentFinish] | List[AgentAction | AgentFinish]:
        ...

    @overload
    async def agenerate(
        self,
        agent: BaseSingleActionAgent,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> List[AgentAction | AgentFinish]:
        ...

    @overload
    async def agenerate(
        self,
        agent: BaseMultiActionAgent,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> List[List[AgentAction] | AgentFinish]:
        ...

    @abstractmethod
    async def agenerate(
        self,
        agent: BaseSingleActionAgent | BaseMultiActionAgent,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> List[List[AgentAction] | AgentFinish] | List[AgentAction | AgentFinish]:
        ...


class AgentThoughtGenerator(BaseThoughtGenerator):
    @overload
    def generate(
        self,
        agent: BaseSingleActionAgent,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[AgentAction | AgentFinish]:
        ...

    @overload
    def generate(
        self,
        agent: BaseMultiActionAgent,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[List[AgentAction] | AgentFinish]:
        ...

    def generate(
        self,
        agent: BaseSingleActionAgent | BaseMultiActionAgent,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[List[AgentAction] | AgentFinish] | List[AgentAction | AgentFinish]:
        # sample: `max_num_thoughts` i.i.d. requests
        if self.generation_mode == "sample":
            results: List[List[AgentAction] | AgentFinish] | List[AgentAction | AgentFinish] = []
            for _ in range(max_num_thoughts):
                cur_result = agent.plan(
                    intermediate_steps=trajectory,
                    callbacks=run_manager,
                    previous_thoughts=results,
                    **inputs,
                )
                # TODO: how to fix mypy warning properly here?
                results.append(cur_result)  # type: ignore[arg-type]

            return results

        raise ValueError("Current thought generation mode is not supported yet.")

    @overload
    async def agenerate(
        self,
        agent: BaseSingleActionAgent,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> List[AgentAction | AgentFinish]:
        ...

    @overload
    async def agenerate(
        self,
        agent: BaseMultiActionAgent,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> List[List[AgentAction] | AgentFinish]:
        ...

    async def agenerate(
        self,
        agent: BaseSingleActionAgent | BaseMultiActionAgent,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> List[List[AgentAction] | AgentFinish] | List[AgentAction | AgentFinish]:
        # sample: `max_num_thoughts` i.i.d. requests
        if self.generation_mode == "sample":
            # TODO: don't repeat suggestions in async version?
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

        raise ValueError("Current thought generation mode is not supported yet.")
