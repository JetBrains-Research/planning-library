from typing import Optional, Tuple

from langchain_core.callbacks import CallbackManager, AsyncCallbackManager

from ..base_component import BaseComponent, InputType, OutputType


class EvaluatorComponent(BaseComponent[InputType, bool]):
    def __init__(
        self,
        backbone: BaseComponent[InputType, OutputType],
        judge: BaseComponent[Tuple[InputType, OutputType], bool],
    ):
        self.backbone = backbone
        self.judge = judge

    def invoke(
        self,
        inputs: InputType,
        run_manager: Optional[CallbackManager] = None,
    ) -> bool:
        backbone_output = self.backbone.invoke(inputs, run_manager)
        should_continue = self.judge.invoke((inputs, backbone_output), run_manager)
        return should_continue

    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> bool:
        backbone_output = await self.backbone.ainvoke(inputs, run_manager)
        should_continue = await self.judge.ainvoke(
            (inputs, backbone_output), run_manager
        )
        return should_continue
