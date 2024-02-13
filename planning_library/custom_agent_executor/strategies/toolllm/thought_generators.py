from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple

from langchain_core.agents import AgentAction
from langchain_core.callbacks import CallbackManager
from langchain_core.runnables import Runnable


class BaseThoughtGenerator(ABC):
    @abstractmethod
    def generate_thoughts(
        self,
        inputs: Dict[str, str],
        current_state: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[str]:
        ...


class LLMThoughtGenerator(BaseThoughtGenerator):
    def __init__(
        self,
        llm_chain: Runnable[Dict[str, Any], str | List[str]],
        thought_generation_mode: Literal["sample", "propose"],
    ):
        self.llm_chain = llm_chain
        self.thought_generation_mode = thought_generation_mode

    def generate_thoughts(
        self,
        inputs: Dict[str, str],
        current_state: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[str]:
        # sample k i.i.d. thoughts
        if self.thought_generation_mode == "sample":
            # TODO: can be async?
            sample_results = []
            for _ in range(max_num_thoughts):
                cur_result = self.llm_chain.invoke(
                    {"inputs": inputs, "max_num_thoughts": max_num_thoughts, "current_state": current_state},
                    {"callbacks": run_manager} if run_manager else {},
                )
                assert isinstance(
                    cur_result, str
                ), "In sample mode, thought generator must return a single thought after a single call."
                sample_results.append(cur_result)
            return sample_results

        # propose k thoughts via a single call
        if self.thought_generation_mode == "propose":
            propose_results = self.llm_chain.invoke(
                {"inputs": inputs, "max_num_thoughts": max_num_thoughts, "current_state": current_state},
                {"callbacks": run_manager} if run_manager else {},
            )
            assert isinstance(
                propose_results, list
            ), "In propose mode, thought generator must return a list of thought after a single call."
            return propose_results

        raise ValueError(f"Unknown thought generation mode {self.thought_generation_mode}")
