from ..base_component import BaseComponent
from typing import Optional
from langchain_core.callbacks import CallbackManager, AsyncCallbackManager
from planning_library.components.base_component import InputType


class LeqThresholdJudge(BaseComponent[InputType, bool]):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def invoke(
        self,
        inputs: InputType,
        run_manager: Optional[CallbackManager] = None,
    ) -> bool:
        return inputs["backbone_output"] <= self.threshold

    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> bool:
        return inputs["backbone_output"] <= self.threshold


class GeqThresholdJudge(BaseComponent[InputType, bool]):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def invoke(
        self,
        inputs: InputType,
        run_manager: Optional[CallbackManager] = None,
    ) -> bool:
        return inputs["backbone_output"] >= self.threshold

    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> bool:
        return inputs["backbone_output"] >= self.threshold
