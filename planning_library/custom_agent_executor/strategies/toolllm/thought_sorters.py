from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager


class BaseThoughtSorter(ABC):
    @abstractmethod
    def sort_thoughts(
        self,
        inputs: Dict[str, str],
        current_state: List[Tuple[AgentAction, str]],
        thoughts: List[Union[AgentAction, AgentFinish]],
        run_manager: Optional[CallbackManager] = None,
    ) -> List[Union[AgentAction, AgentFinish]]:
        ...
