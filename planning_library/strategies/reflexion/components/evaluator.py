from planning_library.components import RunnableComponent
from planning_library.components.evaluation import EvaluatorComponent
from typing import Tuple, Dict, Any, List, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from textwrap import dedent
from planning_library.primitives.output_parsers import SimpleEvaluateOutputParser
from planning_library.components.evaluation import ThresholdJudge
from typing_extensions import TypedDict


class ReflexionEvaluatorInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[Tuple[AgentAction, str]]
    agent_outcome: AgentFinish


class ReflexionEvaluator(EvaluatorComponent[ReflexionEvaluatorInput]):
    @staticmethod
    def _create_prompt(
        system_message: Optional[str], user_message: str
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
                     Take your time and comment your decision, but make sure to always output either 0 or 1 in the end, where 0 would mean 'the episode ended in failure' and 1 would mean 'the episode ended in success'. 
                     Use the following format: [[number]].

                     Your verdict:
                     """),
                ),
            ]
        )

    @staticmethod
    def _create_output_parser() -> BaseOutputParser[float]:
        return SimpleEvaluateOutputParser()

    @classmethod
    def create_from_runnable(
        cls, runnable: Runnable[ReflexionEvaluatorInput, float], threshold: float
    ) -> "ReflexionEvaluator":
        backbone = RunnableComponent(runnable=runnable)
        judge = ThresholdJudge(threshold)
        return cls(backbone=backbone, judge=judge)

    @classmethod
    def create_from_prompt_and_llm(
        cls,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
        threshold: float,
        output_parser: Optional[BaseOutputParser[float]],
    ) -> "ReflexionEvaluator":
        if output_parser is None:
            output_parser = cls._create_output_parser()
        runnable: Runnable = prompt | llm | output_parser
        return cls.create_from_runnable(runnable=runnable, threshold=threshold)

    @classmethod
    def create(
        cls,
        llm: BaseChatModel,
        threshold: float,
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        output_parser: Optional[BaseOutputParser[float]] = None,
    ) -> "ReflexionEvaluator":
        if prompt is None:
            if user_message is None:
                raise ValueError(
                    "Either `prompt` or `user_message` are required to create an agent."
                )
            prompt = cls._create_prompt(
                system_message=system_message, user_message=user_message
            )

        missing_vars = set(ReflexionEvaluatorInput.__annotations__).difference(
            prompt.input_variables
        )
        if missing_vars:
            raise ValueError(f"Prompt missing required variables: {missing_vars}")

        return cls.create_from_prompt_and_llm(
            llm=llm, threshold=threshold, prompt=prompt, output_parser=output_parser
        )
