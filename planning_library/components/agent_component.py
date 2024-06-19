from __future__ import annotations

from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Union

from langchain.agents.agent import BaseMultiActionAgent, BaseSingleActionAgent, RunnableAgent, RunnableMultiActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import AsyncCallbackManager, CallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool

from planning_library.function_calling_parsers import (
    BaseFunctionCallingMultiActionParser,
    BaseFunctionCallingSingleActionParser,
    ParserRegistry,
)
from planning_library.utils import (
    convert_runnable_to_agent,
)

from .base_component import BaseComponent, InputType


class AgentFactory:
    @staticmethod
    def create_agent(
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        prompt: ChatPromptTemplate,
        parser: Optional[
            Union[
                BaseFunctionCallingSingleActionParser,
                BaseFunctionCallingMultiActionParser,
            ]
        ] = None,
        parser_name: Optional[str] = None,
    ) -> Union[RunnableAgent, RunnableMultiActionAgent]:
        if parser is None:
            if parser_name is None:
                raise ValueError("Either parser or parser_name should be provided to instantiate an agent.")
            parser = ParserRegistry.get_parser(parser_name)

        llm_with_tools = parser.prepare_llm(llm=llm, tools=tools)

        runnable: Runnable = RunnableLambda(parser.format_inputs) | prompt | llm_with_tools | parser.output_parser
        agent = convert_runnable_to_agent(runnable)
        return agent


class AgentComponent(BaseComponent[InputType, Union[List[AgentAction], AgentAction, AgentFinish]]):
    def __init__(
        self,
        agent: BaseSingleActionAgent | BaseMultiActionAgent,
        flatten_inputs: bool = True,
    ):
        self.agent = agent
        if flatten_inputs:
            self.add_input_preprocessing(
                lambda x: {
                    **x["inputs"],
                    **{key: x[key] for key in x if key != "inputs"},
                }
            )

    def add_input_preprocessing(
        self,
        preprocess: Callable[[InputType], Dict],
        apreprocess: Optional[Callable[[InputType], Awaitable[Dict]]] = None,
    ) -> None:
        if hasattr(self.agent, "runnable"):
            self.agent.runnable = RunnableLambda(preprocess, afunc=apreprocess) | self.agent.runnable  # type: ignore[reportAttributeAccessIssue]

    def add_output_preprocessing(
        self,
        preprocess: Callable[
            [Union[List[AgentAction], AgentAction, AgentFinish]],
            Union[List[AgentAction], AgentAction, AgentFinish],
        ],
        apreprocess: Optional[
            Callable[
                [Union[List[AgentAction], AgentAction, AgentFinish]],
                Awaitable[Union[List[AgentAction], AgentAction, AgentFinish]],
            ]
        ] = None,
    ) -> None:
        if hasattr(self.agent, "runnable"):
            self.agent.runnable = self.agent.runnable | RunnableLambda(preprocess, afunc=apreprocess)  # type: ignore[reportAttributeAccessIssue]

    def invoke(
        self, inputs: InputType, run_manager: Optional[CallbackManager] = None, **kwargs
    ) -> Union[List[AgentAction], AgentAction, AgentFinish]:
        # TODO: no way to pass name to plan?
        # TODO: intermediate_steps?
        return self.agent.plan(**inputs, callbacks=run_manager)  # type: ignore[reportCallIssue]

    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> Union[List[AgentAction], AgentAction, AgentFinish]:
        # TODO: no way to pass name to plan?
        outputs = await self.agent.aplan(**inputs, callbacks=run_manager)  # type: ignore[reportCallIssue]
        return outputs

    @classmethod
    def create(
        cls,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        parser: Optional[
            Union[
                BaseFunctionCallingSingleActionParser,
                BaseFunctionCallingMultiActionParser,
            ]
        ] = None,
        parser_name: Optional[str] = None,
    ) -> "AgentComponent[InputType]":
        prompt = cls._process_prompt(prompt=prompt, user_message=user_message, system_message=system_message)

        return cls(
            agent=AgentFactory.create_agent(
                llm=llm,
                tools=tools,
                prompt=prompt,
                parser=parser,
                parser_name=parser_name,
            )
        )
