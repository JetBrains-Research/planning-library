from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager


class BaseThoughtGenerator(ABC):
    @abstractmethod
    def generate_thoughts(
        self,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        inputs: Dict[str, str],
        current_state: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[Union[AgentAction, AgentFinish]]:
        ...


class AgentThoughtGenerator(BaseThoughtGenerator):
    def __init__(
        self,
        thought_generation_mode: Literal["sample", "propose"],
    ):
        self.thought_generation_mode = thought_generation_mode

    def generate_thoughts(
        self,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        inputs: Dict[str, str],
        current_state: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[Union[AgentAction, AgentFinish]]:
        # sample k i.i.d. thoughts
        if self.thought_generation_mode == "sample":
            # TODO: can be async?
            sample_results = []
            for _ in range(max_num_thoughts):
                cur_result = agent.plan(
                    current_state,
                    callbacks=run_manager,
                    **inputs,
                )
                assert isinstance(cur_result, AgentAction) or isinstance(
                    cur_result, AgentFinish
                ), "In sample mode, thought generator must return a single thought after a single call."
                sample_results.append(cur_result)
            return sample_results

        # propose k thoughts via a single call
        if self.thought_generation_mode == "propose":
            propose_results = agent.plan(
                current_state,
                callbacks=run_manager,
                max_num_thoughts=max_num_thoughts,
                **inputs,
            )
            assert isinstance(
                propose_results, list
            ), "In propose mode, thought generator must return a list of thought after a single call."
            return propose_results  # type: ignore[return-value]

        raise ValueError(f"Unknown thought generation mode {self.thought_generation_mode}")
