from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish, AgentStep


class BaseThoughtSorter(ABC):
    @abstractmethod
    def sort_thoughts(
        self,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        inputs: Dict[str, str],
        current_state: List[Tuple[AgentAction, str]],
        thoughts: List[Union[AgentAction, AgentFinish]],
    ) -> List[Union[AgentAction, AgentFinish]]:
        ...
