from ..base_component import BaseComponent, InputType
from typing import Tuple, Optional
from langchain_core.callbacks import CallbackManager, AsyncCallbackManager


class ThresholdJudge(BaseComponent[Tuple[InputType, float], bool]):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def invoke(
        self,
        inputs: Tuple[InputType, float],
        run_manager: Optional[CallbackManager],
    ) -> bool:
        return inputs[1] > self.threshold

    async def ainvoke(
        self,
        inputs: Tuple[InputType, float],
        run_manager: Optional[AsyncCallbackManager],
    ) -> bool:
        return inputs[1] > self.threshold
