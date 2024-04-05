from planning_library.components.evaluation import EvaluatorComponent
from typing import Tuple, Dict, Any, List, Optional, Generic, Type
from planning_library.components.base_component import OutputType
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from textwrap import dedent
from planning_library.function_calling_parsers import ParserRegistry
from typing_extensions import TypedDict
from functools import partial


class ReflexionEvaluatorInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[Tuple[AgentAction, str]]
    agent_outcome: AgentFinish


class PreprocessedReflexionEvaluatorInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[BaseMessage]
    agent_outcome: str


class ReflexionEvaluator(
    Generic[OutputType], EvaluatorComponent[ReflexionEvaluatorInput, OutputType]
):
    required_prompt_input_vars = set(ReflexionEvaluatorInput.__annotations__)

    @classmethod
    def _create_default_prompt(
        cls, system_message: Optional[str], user_message: str
    ) -> ChatPromptTemplate:
        if system_message is None:
            system_message = "You are an advanced reasoning assistant that judges whether the episodes result in success or failure."

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    user_message,
                ),
                ("human", "Inputs: {inputs}"),
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

    @staticmethod
    def _preprocess_input(
        inputs: ReflexionEvaluatorInput, parser, parser_name
    ) -> PreprocessedReflexionEvaluatorInput:
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
        cls: Type["ReflexionEvaluator"],
        llm: BaseChatModel,
        threshold: float,
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        output_parser: Optional[BaseOutputParser[float]] = None,
        parser=None,
        parser_name=None,
    ) -> "ReflexionEvaluator[float]":
        evaluator = cls.create_threshold_evaluator(
            llm=llm,
            threshold=threshold,
            threshold_mode="leq",
            prompt=prompt,
            user_message=user_message,
            system_message=system_message,
            output_parser=output_parser,
        )

        if hasattr(evaluator.backbone, "runnable"):
            evaluator.backbone.runnable = (
                RunnableLambda(
                    partial(
                        cls._preprocess_input, parser=parser, parser_name=parser_name
                    )
                )
                | evaluator.backbone.runnable
            )
        # TODO: fix typing here
        return evaluator
