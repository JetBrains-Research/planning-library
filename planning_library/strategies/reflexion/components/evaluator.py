from __future__ import annotations

from textwrap import dedent
from typing import Any, Dict, Generic, List, Optional, Tuple, Type

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing_extensions import TypedDict

from planning_library.components.base_component import OutputType
from planning_library.components.evaluation import EvaluatorComponent
from planning_library.function_calling_parsers import (
    BaseFunctionCallingMultiActionParser,
    BaseFunctionCallingSingleActionParser,
    ParserRegistry,
)


class ReflexionEvaluatorInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[Tuple[AgentAction, str]]
    agent_outcome: AgentFinish


class PreprocessedReflexionEvaluatorInput(TypedDict):
    intermediate_steps: List[BaseMessage]
    agent_outcome: str


class ReflexionEvaluator(Generic[OutputType], EvaluatorComponent[ReflexionEvaluatorInput, OutputType]):
    name = "Evaluator"

    required_prompt_input_vars = set(ReflexionEvaluatorInput.__annotations__) - {"inputs"}

    @classmethod
    def _create_default_prompt(cls, system_message: Optional[str], user_message: str, **kwargs) -> ChatPromptTemplate:
        if system_message is None:
            system_message = (
                "You are an advanced reasoning assistant that judges whether the episodes result in success or failure."
            )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    user_message,
                ),
                MessagesPlaceholder("intermediate_steps"),
                ("human", "Answer: {agent_outcome}"),
                (
                    "human",
                    dedent("""
                     Take your time and comment your decision, but make sure to always output number between 0 and 1 in the end, where 0 would mean 'the episode ended in failure' and 1 would mean 'the episode ended in success'. 
                     Use the following format: [[number]].

                     Your verdict:
                     """),
                ),
            ]
        )

    @classmethod
    def create(
        cls: Type["ReflexionEvaluator"],
        llm: BaseChatModel,
        threshold: float,
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        output_parser: Optional[BaseOutputParser[float]] = None,
        parser: Optional[BaseFunctionCallingSingleActionParser | BaseFunctionCallingMultiActionParser] = None,
        parser_name: Optional[str] = None,
    ) -> "ReflexionEvaluator[float]":
        def _preprocess_input(
            inputs: ReflexionEvaluatorInput,
        ) -> Dict:
            # TODO: figure out typing here
            nonlocal parser, parser_name
            if parser is None:
                assert parser_name is not None
                parser = ParserRegistry.get_parser(parser_name)

            preprocessed_inputs = parser.format_inputs(inputs)
            return {
                **preprocessed_inputs["inputs"],
                "agent_outcome": preprocessed_inputs["agent_outcome"].return_values["output"],  # type: ignore[typeddict-item]
                "intermediate_steps": preprocessed_inputs["agent_scratchpad"],
            }

        # TODO: fix typing here
        evaluator: ReflexionEvaluator = cls.create_threshold_evaluator(  # type: ignore[assignment]
            llm=llm,
            threshold=threshold,
            threshold_mode="leq",
            prompt=prompt,
            user_message=user_message,
            system_message=system_message,
            output_parser=output_parser,
        )

        evaluator.add_input_preprocessing(_preprocess_input)

        return evaluator
