from typing import Awaitable, Callable, Dict, Optional

from langchain_core.callbacks import AsyncCallbackManager, CallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from .base_component import BaseComponent, InputType, OutputType


class RunnableComponent(BaseComponent[InputType, OutputType]):
    def __init__(self, runnable: Runnable[InputType, OutputType]):
        self.runnable = runnable

    @classmethod
    def create_from_steps(
        cls,
        llm: BaseChatModel,
        output_parser: Optional[BaseOutputParser] = None,
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> "RunnableComponent":
        prompt = cls._process_prompt(prompt=prompt, user_message=user_message, system_message=system_message)
        runnable = prompt | llm
        if output_parser is not None:
            runnable = runnable | output_parser
        return RunnableComponent(runnable)

    def add_input_preprocessing(
        self,
        preprocess: Callable[[InputType], Dict],
        apreprocess: Optional[Callable[[InputType], Awaitable[Dict]]] = None,
    ) -> None:
        self.runnable = RunnableLambda(preprocess, afunc=apreprocess) | self.runnable

    def add_output_preprocessing(
        self,
        preprocess: Callable[[OutputType], OutputType],
        apreprocess: Optional[Callable[[OutputType], Awaitable[OutputType]]] = None,
    ) -> None:
        self.runnable = self.runnable | RunnableLambda(preprocess, afunc=apreprocess)

    def invoke(self, inputs: InputType, run_manager: Optional[CallbackManager] = None, **kwargs) -> OutputType:
        config = kwargs
        if "callbacks" not in config and run_manager:
            config["callbacks"] = run_manager

        if "run_name" not in config and self.name:
            config["run_name"] = self.name

        outputs = self.runnable.invoke(
            inputs,
            config=config,  # type: ignore[arg-type]
        )
        return outputs

    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> OutputType:
        config = kwargs
        if "callbacks" not in config and run_manager:
            config["callbacks"] = run_manager

        if "run_name" not in config and self.name:
            config["run_name"] = self.name

        outputs = await self.runnable.ainvoke(
            inputs,
            config=config,  # type: ignore[arg-type]
        )
        return outputs
