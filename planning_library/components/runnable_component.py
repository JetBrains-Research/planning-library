from typing import Optional
from langchain_core.runnables import Runnable
from langchain_core.callbacks import CallbackManager, AsyncCallbackManager
from .base_component import InputType, OutputType, BaseComponent


class RunnableComponent(BaseComponent[InputType, OutputType]):
    def __init__(self, runnable: Runnable[InputType, OutputType]):
        self.runnable = runnable

    def invoke(
        self,
        inputs: InputType,
        run_manager: Optional[CallbackManager] = None,
    ) -> OutputType:
        outputs = self.runnable.invoke(
            inputs,
            config={"callbacks": run_manager} if run_manager else {},
        )
        return outputs

    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> OutputType:
        outputs = await self.runnable.ainvoke(
            inputs,
            config={"callbacks": run_manager} if run_manager else {},
        )
        return outputs
