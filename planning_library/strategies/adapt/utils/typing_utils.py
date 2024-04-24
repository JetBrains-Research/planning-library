from typing import List, Literal, Dict, Any, Tuple
from langchain_core.agents import AgentAction, AgentFinish
from typing_extensions import TypedDict
from langchain_core.pydantic_v1 import BaseModel, Field


class ADaPTPlannerInput(TypedDict):
    inputs: Dict[str, Any]
    executor_agent_outcome: AgentFinish
    executor_intermediate_steps: List[Tuple[AgentAction, str]]


class ADaPTPlannerOutput(TypedDict):
    subtasks: List[str]
    aggregation_mode: Literal["and", "or"]


class ADaPTPlan(BaseModel):
    subtasks: List[str] = Field(default_factory=list)
    aggregation_mode: Literal["and", "or"] = Field(default="and")

    def clear(self):
        self.subtasks = []
        self.aggregation_mode = "and"
