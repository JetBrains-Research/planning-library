from typing import Optional, Dict, Callable, Awaitable
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.callbacks import CallbackManager, AsyncCallbackManager
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.language_models import BaseChatModel
from .base_component import InputType, OutputType, BaseComponent


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
        prompt = cls._process_prompt(
            prompt=prompt, user_message=user_message, system_message=system_message
        )
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
