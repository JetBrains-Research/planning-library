from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.agents import AgentAction
from langchain_core.callbacks import CallbackManager
from langchain_core.runnables import Runnable


class BaseThoughtEvaluator(ABC):
    @abstractmethod
    def evaluate_thought(
        self,
        inputs: Dict[str, str],
        current_state: List[Tuple[AgentAction, str]],
        next_thought: str,
        run_manager: Optional[CallbackManager] = None,
    ) -> float:
        ...


class LLMThoughtEvaluator(BaseThoughtEvaluator):
    def __init__(self, llm_chain: Runnable[Dict[str, Any], float]):
        self.llm_chain = llm_chain

    def evaluate_thought(
        self,
        inputs: Dict[str, str],
        current_state: List[Tuple[AgentAction, str]],
        next_thought: str,
        run_manager: Optional[CallbackManager] = None,
    ) -> float:
        return self.llm_chain.invoke(
            {"inputs": inputs, "current_state": current_state, "thought": next_thought},
            {"callbacks": run_manager} if run_manager else {},
        )
