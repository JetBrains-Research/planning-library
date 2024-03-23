from typing_extensions import TypedDict
from typing import Dict, Any, List, Literal


class ADaPTTask(TypedDict):
    inputs: Dict[str, Any]
    depth: int


class ADaPTPlan(TypedDict):
    subtasks: List[ADaPTTask]
    logic: Literal["and", "or"]
