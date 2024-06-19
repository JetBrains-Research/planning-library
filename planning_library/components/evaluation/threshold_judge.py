from typing import Optional

from langchain_core.callbacks import AsyncCallbackManager, CallbackManager

from planning_library.components.base_component import InputType

from ..base_component import BaseComponent


class LeqThresholdJudge(BaseComponent[InputType, bool]):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def invoke(self, inputs: InputType, run_manager: Optional[CallbackManager] = None, **kwargs) -> bool:
        return inputs["backbone_output"] <= self.threshold

    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> bool:
        return inputs["backbone_output"] <= self.threshold


class GeqThresholdJudge(BaseComponent[InputType, bool]):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def invoke(self, inputs: InputType, run_manager: Optional[CallbackManager] = None, **kwargs) -> bool:
        return inputs["backbone_output"] >= self.threshold

    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> bool:
        return inputs["backbone_output"] >= self.threshold
