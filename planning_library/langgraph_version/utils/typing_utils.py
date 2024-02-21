import operator
from typing import Annotated, Any, List, Sequence, Tuple, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish


class AgentState(TypedDict):
    inputs: str
    agent_outcome: Union[AgentAction, AgentFinish, None]
    evaluator_score: Any
    self_reflections: Annotated[Sequence[str], operator.add]
    intermediate_steps: Annotated[List[Tuple[AgentAction, str]], operator.add]
    iteration: int
