from langchain_core.callbacks import (
    CallbackManager,
    AsyncCallbackManager,
)
from typing import Optional, Dict, Any
from planning_library.strategies import BaseCustomStrategy
from .base_component import InputType, BaseComponent


class StrategyComponent(BaseComponent[InputType, Dict[str, Any]]):
    def __init__(self, strategy: BaseCustomStrategy):
        self.strategy = strategy

    def invoke(
        self,
        inputs: InputType,
        run_manager: Optional[CallbackManager] = None,
    ) -> Dict[str, Any]:
        outputs = self.strategy.invoke(
            inputs,
            config={"callbacks": run_manager} if run_manager else {},
        )
        return outputs

    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> Dict[str, Any]:
        outputs = await self.strategy.ainvoke(
            inputs,
            config={"callbacks": run_manager} if run_manager else {},
        )
        return outputs
