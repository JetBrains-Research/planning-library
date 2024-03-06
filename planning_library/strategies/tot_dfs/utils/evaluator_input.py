from typing import Dict, List, Optional, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from typing_extensions import TypedDict


class EvaluatorInput(TypedDict):
    inputs: Dict[str, str]
    trajectory: List[Tuple[AgentAction, str]]
    next_thought: Union[List[AgentAction], AgentAction, AgentFinish]
    observation: Optional[Union[List[AgentStep], AgentStep]]
