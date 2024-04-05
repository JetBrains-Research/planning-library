from planning_library.components import RunnableComponent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from textwrap import dedent
from typing import Optional, Sequence, Tuple, Dict, Any, List
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from typing_extensions import TypedDict
from planning_library.function_calling_parsers import ParserRegistry
from functools import partial


class ReflexionSelfReflectionInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[Tuple[AgentAction, str]]
    agent_outcome: AgentFinish


class PreprocessedReflexionSelfReflectionInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[BaseMessage]
    agent_outcome: str


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

    @staticmethod
    def _preprocess_input(
        inputs: ReflexionSelfReflectionInput,
        parser,
        parser_name,
    ) -> PreprocessedReflexionSelfReflectionInput:
        if parser is None:
            assert parser_name is not None
            parser = ParserRegistry.get_parser(parser_name)

        preprocessed_inputs = parser.format_inputs(inputs)
        return {
            "inputs": preprocessed_inputs["inputs"],
            "agent_outcome": preprocessed_inputs["agent_outcome"].return_values[
                "output"
            ],
            "intermediate_steps": preprocessed_inputs["agent_scratchpad"],
        }

    @classmethod
    def create(
        cls,
        llm: BaseChatModel,
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        parser=None,
        parser_name=None,
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

        runnable: Runnable = (
            RunnableLambda(
                partial(cls._preprocess_input, parser=parser, parser_name=parser_name)
            )
            | prompt
            | llm
        )

        return ReflexionSelfReflection(runnable=runnable)
