from typing import Dict, List, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish
from typing_extensions import TypedDict


class EvaluatorInput(TypedDict):
    inputs: Dict[str, str]
    trajectory: List[Tuple[AgentAction, str]]
    next_thought: Union[List[AgentAction], AgentAction, AgentFinish]
