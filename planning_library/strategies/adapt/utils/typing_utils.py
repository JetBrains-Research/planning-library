from typing import List, Literal
from langchain_core.pydantic_v1 import BaseModel, Field


class ADaPTPlan(BaseModel):
    subtasks: List[str] = Field(default_factory=list)
    aggregation_mode: Literal["and", "or"] = Field(default="and")

    def clear(self):
        self.subtasks = []
        self.aggregation_mode = "and"
