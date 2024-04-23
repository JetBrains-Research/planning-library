from __future__ import annotations
from planning_library.components.evaluation import EvaluatorComponent
from typing import Tuple, Dict, Any, List, Optional, Generic, Type, Union
from planning_library.components.base_component import OutputType
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from planning_library.utils import (
    format_thought,
)
from textwrap import dedent
from planning_library.function_calling_parsers import (
    ParserRegistry,
    BaseFunctionCallingMultiActionParser,
    BaseFunctionCallingSingleActionParser,
)
from typing_extensions import TypedDict
from dataclasses import dataclass


@dataclass
class ThoughtEvaluatorConfig:
    value_threshold: float

    runnable: Optional[Runnable] = None

    prompt: Optional[ChatPromptTemplate] = None
    user_message: Optional[str] = None
    system_message: Optional[str] = None

    llm: Optional[BaseChatModel] = None

    parser: Optional[
        Union[
            BaseFunctionCallingSingleActionParser,
            BaseFunctionCallingMultiActionParser,
        ]
    ] = None
    parser_name: Optional[str] = None

    output_parser: Optional[BaseOutputParser[float]] = None


class ThoughtEvaluatorInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[Tuple[AgentAction, str]]
    next_thought: List[AgentAction] | AgentAction | AgentFinish


class ThoughtEvaluator(
    Generic[OutputType], EvaluatorComponent[ThoughtEvaluatorInput, OutputType]
):
    name = "Evaluate Thoughts"

    required_prompt_input_vars = set(ThoughtEvaluatorInput.__annotations__) - {"inputs"}

    @classmethod
    def _create_default_prompt(
        cls, system_message: Optional[str], user_message: str
    ) -> ChatPromptTemplate:
        if system_message is None:
            system_message = (
                "You are an advanced reasoning assistant that judges the plausability of "
                "steps suggested for solving complex reasoning tasks."
            )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    user_message,
                ),
                MessagesPlaceholder("intermediate_steps"),
                ("human", "Here is the proposed next step:" ""),
                MessagesPlaceholder("next_thought"),
                (
                    "human",
                    dedent("""
                     Your goal is to judge whether this proposal should be followed or discarded, 
                     how likely it is to lead to the success.
                     
                     Take your time and comment your decision, 
                     but make sure to always output number between 0 and 1 in the end, 
                     where 0 would mean 'the proposed action is incorrect and/or very unlikely to lead to success' 
                     and 1 would mean 'the proposed action is correct and very likely to lead to success'. 
                     
                     ALWAYS use the following format and add the number in the end of your answer: [[number]].

                     Your verdict:
                     """),
                ),
            ]
        )

    @classmethod
    def create(
        cls: Type["ThoughtEvaluator"],
        llm: BaseChatModel,
        threshold: float,
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        output_parser: Optional[BaseOutputParser[float]] = None,
        parser: Optional[
            BaseFunctionCallingSingleActionParser | BaseFunctionCallingMultiActionParser
        ] = None,
        parser_name: Optional[str] = None,
    ) -> "ThoughtEvaluator[float]":
        def _preprocess_input(
            inputs: ThoughtEvaluatorInput,
        ) -> Dict:
            # TODO: figure out typing here
            nonlocal parser, parser_name
            if parser is None:
                assert parser_name is not None
                parser = ParserRegistry.get_parser(parser_name)

            intermediate_steps = parser.format_inputs(inputs)["agent_scratchpad"]

            return {
                **inputs["inputs"],
                "next_thought": format_thought(inputs["next_thought"]),
                "intermediate_steps": intermediate_steps,
            }

        # TODO: fix typing here
        evaluator: ThoughtEvaluator = cls.create_threshold_evaluator(  # type: ignore[assignment]
            llm=llm,
            threshold=threshold,
            threshold_mode="geq",
            prompt=prompt,
            user_message=user_message,
            system_message=system_message,
            output_parser=output_parser,
        )

        evaluator.add_input_preprocessing(_preprocess_input)

        return evaluator

    @classmethod
    def create_from_config(cls, config: ThoughtEvaluatorConfig) -> ThoughtEvaluator:
        if config.runnable is not None:
            evaluator: ThoughtEvaluator = cls.create_threshold_evaluator_from_runnable(  # type: ignore[assignment]
                runnable=config.runnable,
                threshold=config.value_threshold,
                threshold_mode="geq",
            )
            return evaluator

        if config.llm is None:
            raise ValueError("`llm` must be provided when `runnable` is None.")

        return cls.create(
            llm=config.llm,
            prompt=config.prompt,
            user_message=config.user_message,
            system_message=config.system_message,
            threshold=config.value_threshold,
            output_parser=config.output_parser,
            parser=config.parser,
            parser_name=config.parser_name,
        )
