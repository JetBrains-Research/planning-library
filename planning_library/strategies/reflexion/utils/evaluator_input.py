from typing import Any, Dict, List, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish
from typing_extensions import TypedDict


class BaseEvaluatorInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[Tuple[AgentAction, str]]


class ReflexionEvaluatorInput(BaseEvaluatorInput):
    agent_outcome: Union[List[AgentAction], AgentAction, AgentFinish]
