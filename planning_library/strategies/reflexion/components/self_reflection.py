from planning_library.components import RunnableComponent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from textwrap import dedent
from typing import Optional, Sequence, Tuple, Dict, Any, List
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from typing_extensions import TypedDict


class ReflexionSelfReflectionInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[Tuple[AgentAction, str]]
    agent_outcome: AgentFinish


class ReflexionSelfReflection(
    RunnableComponent[ReflexionSelfReflectionInput, Sequence[BaseMessage]]
):
    @staticmethod
    def _create_prompt(
        system_message: Optional[str], user_message: str
    ) -> ChatPromptTemplate:
        if system_message is None:
            system_message = "You are an advanced reasoning agent that can self-reflect on their shortcomings when solving reasoning tasks."

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    user_message,
                ),
                ("human", "Inputs: {inputs}"),
                MessagesPlaceholder("intermediate_steps"),
                ("human", "Final Outcome: {agent_outcome}"),
                (
                    "human",
                    dedent("""
                     In this trial, you were unsuccessful. 
                     In a few sentences, diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same shortcomings. 
                     Use complete sentences.""
                     """),
                ),
            ]
        )

    @classmethod
    def create(
        cls,
        llm: BaseChatModel,
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> "ReflexionSelfReflection":
        if prompt is None:
            if user_message is None:
                raise ValueError(
                    "Either `prompt` or `user_message` are required to create an agent."
                )
            prompt = cls._create_prompt(
                system_message=system_message, user_message=user_message
            )

        missing_vars = set(ReflexionSelfReflectionInput.__annotations__).difference(
            prompt.input_variables
        )
        if missing_vars:
            raise ValueError(f"Prompt missing required variables: {missing_vars}")

        runnable: Runnable = prompt | llm

        return ReflexionSelfReflection(runnable=runnable)
