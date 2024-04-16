from typing import Dict, Any
from planning_library.strategies import BaseCustomStrategy
from .base_component import InputType
from .runnable_component import RunnableComponent


class StrategyComponent(RunnableComponent[InputType, Dict[str, Any]]):
    def __init__(self, strategy: BaseCustomStrategy):
        # TODO: typing: how to show that it's a runnable?
        super().__init__(strategy)  # type: ignore[arg-type]
