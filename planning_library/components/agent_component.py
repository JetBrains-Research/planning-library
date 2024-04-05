from __future__ import annotations

from typing import Optional, List, Union, Sequence
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager, AsyncCallbackManager
from .base_component import InputType, BaseComponent
from langchain.agents.agent import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from planning_library.utils import (
    convert_runnable_to_agent,
)
from langchain.agents.agent import RunnableAgent, RunnableMultiActionAgent
from langchain_core.runnables import RunnableLambda, Runnable
from planning_library.function_calling_parsers import (
    BaseFunctionCallingSingleActionParser,
    BaseFunctionCallingMultiActionParser,
    ParserRegistry,
)


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
                raise ValueError(
                    "Either parser or parser_name should be provided to instantiate an agent."
                )
            parser = ParserRegistry.get_parser(parser_name)

        llm_with_tools = llm.bind(tools=[parser.prepare_tool(tool) for tool in tools])
        runnable: Runnable = (
            RunnableLambda(parser.format_inputs)
            | prompt
            | llm_with_tools
            | parser.output_parser
        )
        agent = convert_runnable_to_agent(runnable)
        return agent


class AgentComponent(
    BaseComponent[InputType, Union[List[AgentAction], AgentAction, AgentFinish]]
):
    def __init__(self, agent: BaseSingleActionAgent | BaseMultiActionAgent):
        self.agent = agent

    def invoke(
        self,
        inputs: InputType,
        run_manager: Optional[CallbackManager] = None,
    ) -> Union[List[AgentAction], AgentAction, AgentFinish]:
        return self.agent.plan(**inputs, callbacks=run_manager)

    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> Union[List[AgentAction], AgentAction, AgentFinish]:
        outputs = await self.agent.aplan(**inputs, callbacks=run_manager)
        return outputs

    @classmethod
    def create_agent(
        cls,
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
    ) -> "AgentComponent[InputType]":
        return cls(
            agent=AgentFactory.create_agent(
                llm=llm,
                tools=tools,
                prompt=prompt,
                parser=parser,
                parser_name=parser_name,
            )
        )
