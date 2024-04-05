from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from textwrap import dedent
from typing import Sequence, Optional, Union
from planning_library.components import AgentComponent
from typing import Tuple, Dict, Any, List
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from planning_library.function_calling_parsers import (
    BaseFunctionCallingSingleActionParser,
    BaseFunctionCallingMultiActionParser,
)


class ReflexionActorInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[Tuple[AgentAction, str]]
    self_reflections: List[BaseMessage]


class ReflexionActor(AgentComponent[ReflexionActorInput]):
    @staticmethod
    def _create_default_prompt(
        system_message: Optional[str], user_message: str
    ) -> ChatPromptTemplate:
        if system_message is None:
            system_message = "You are an advanced reasoning agent that can improve based on self-reflection."

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    user_message,
                ),
                (
                    "human",
                    dedent("""
                        This might be not your first attempt. 
                        In this case, you will find self-reflective thoughts below. 
                        Make sure to pay extra attention to them, as they aim to identify and mitigate the exact shortcomings that led to failure in previous trials. 
                        """),
                ),
                MessagesPlaceholder("self_reflections"),
                (
                    "human",
                    "Good luck!",
                ),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

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
    ) -> ReflexionActor:
        if prompt is None:
            if user_message is None:
                raise ValueError(
                    "Either `prompt` or `user_message` are required to create an agent."
                )
            prompt = cls._create_default_prompt(
                system_message=system_message, user_message=user_message
            )

        missing_vars = (
            set(ReflexionActorInput.__annotations__) - {"intermediate_steps"}
            | {"agent_scratchpad"}
        ).difference(prompt.input_variables)
        if missing_vars:
            raise ValueError(f"Prompt missing required variables: {missing_vars}")

        return cls.create_agent(
            llm=llm, tools=tools, prompt=prompt, parser=parser, parser_name=parser_name
        )
