from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish, AgentStep


class BaseThoughtGenerator(ABC):
    @abstractmethod
    def generate_thoughts(
        self,
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        inputs: Dict[str, str],
        current_state: List[Tuple[AgentAction, str]],
        max_num_thoughts: int,
    ) -> List[Union[AgentAction, AgentFinish]]:
        ...
