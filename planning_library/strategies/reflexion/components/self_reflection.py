from __future__ import annotations
from planning_library.components import RunnableComponent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel
from textwrap import dedent
from typing import Optional, Sequence, Tuple, Dict, Any, List, Type
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from planning_library.function_calling_parsers import (
    ParserRegistry,
    BaseFunctionCallingMultiActionParser,
    BaseFunctionCallingSingleActionParser,
)


class ReflexionSelfReflectionInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[Tuple[AgentAction, str]]
    agent_outcome: AgentFinish


class PreprocessedReflexionSelfReflectionInput(TypedDict):
    intermediate_steps: List[BaseMessage]
    agent_outcome: str


class ReflexionSelfReflection(
    RunnableComponent[ReflexionSelfReflectionInput, Sequence[BaseMessage]]
):
    name = "Self-Reflection"

    required_prompt_input_vars = set(ReflexionSelfReflectionInput.__annotations__) - {
        "inputs"
    }

    @classmethod
    def _create_default_prompt(
        cls, system_message: Optional[str], user_message: str
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
        cls: Type["ReflexionSelfReflection"],
        llm: BaseChatModel,
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        parser: Optional[
            BaseFunctionCallingSingleActionParser | BaseFunctionCallingMultiActionParser
        ] = None,
        parser_name: Optional[str] = None,
    ) -> "ReflexionSelfReflection":
        def _preprocess_input(
            inputs: ReflexionSelfReflectionInput,
        ) -> Dict:
            nonlocal parser, parser_name
            if parser is None:
                assert parser_name is not None
                parser = ParserRegistry.get_parser(parser_name)

            preprocessed_inputs = parser.format_inputs(inputs)
            return {
                **preprocessed_inputs["inputs"],
                "agent_outcome": preprocessed_inputs["agent_outcome"].return_values[  # type: ignore[typeddict-item]
                    "output"
                ],
                "intermediate_steps": preprocessed_inputs["agent_scratchpad"],
            }

        prompt = cls._process_prompt(
            prompt=prompt, user_message=user_message, system_message=system_message
        )

        # TODO: figure out typing here
        self_reflection: ReflexionSelfReflection = cls.create_from_steps(  # type: ignore[assignment]
            prompt=prompt, llm=llm
        )
        self_reflection.add_input_preprocessing(_preprocess_input)
        return self_reflection
