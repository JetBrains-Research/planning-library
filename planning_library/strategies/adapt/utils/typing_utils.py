from typing_extensions import TypedDict
from typing import Dict, Any, List, Literal


class ADaPTTask(TypedDict):
    inputs: Dict[str, Any]
    depth: int


class InitialADaPTPlan(TypedDict):
    subtasks: List[Dict[str, Any]]
    logic: Literal["and", "or"]


class ADaPTPlan(TypedDict):
    subtasks: List[ADaPTTask]
    logic: Literal["and", "or"]
