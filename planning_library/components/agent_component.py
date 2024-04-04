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
    create_custom_agent,
    CustomAgentComponents,
    convert_runnable_to_agent,
)
from langchain.agents import create_openai_functions_agent
from langchain.agents import create_openai_tools_agent


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
        agent_type: str,
        prompt: ChatPromptTemplate,
        components: Optional[CustomAgentComponents] = None,
    ) -> "AgentComponent[InputType]":
        if agent_type == "openai_tools":
            agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

        elif agent_type == "openai_functions":
            agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)

        elif agent_type == "custom":
            if components is None:
                raise ValueError("`components` is required to create a custom agent.")
            agent = create_custom_agent(
                llm=llm, tools=tools, prompt=prompt, components=components
            )
        else:
            raise ValueError(f"Agent type {agent_type} is currently not supported.")

        return cls(agent=convert_runnable_to_agent(agent))
