from typing import Optional, Dict, Generic, Type, Callable, Awaitable

from langchain_core.callbacks import CallbackManager, AsyncCallbackManager
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from ..base_component import BaseComponent, InputType, OutputType
from planning_library.primitives.output_parsers import SimpleEvaluateOutputParser
from .threshold_judge import LeqThresholdJudge, GeqThresholdJudge
from planning_library.components.runnable_component import RunnableComponent


class EvaluatorComponent(
    Generic[InputType, OutputType], BaseComponent[InputType, bool]
):
    def __init__(
        self,
        backbone: BaseComponent[InputType, OutputType],
        judge: BaseComponent[Dict[str, OutputType], bool],
    ):
        self.backbone = backbone
        self.judge = judge

    def add_input_preprocessing(
        self,
        preprocess: Callable[[InputType], Dict],
        apreprocess: Optional[Callable[[InputType], Awaitable[Dict]]] = None,
    ) -> None:
        self.backbone.add_input_preprocessing(preprocess, apreprocess)

    def add_output_preprocessing(
        self,
        preprocess: Callable[[bool], bool],
        apreprocess: Optional[Callable[[bool], Awaitable[bool]]] = None,
    ) -> None:
        self.judge.add_output_preprocessing(preprocess, apreprocess)

    def invoke(
        self,
        inputs: InputType,
        run_manager: Optional[CallbackManager] = None,
    ) -> bool:
        backbone_output = self.backbone.invoke(inputs, run_manager)
        should_continue = self.judge.invoke(
            {"backbone_output": backbone_output}, run_manager
        )
        return should_continue

    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> bool:
        backbone_output = await self.backbone.ainvoke(inputs, run_manager)
        should_continue = await self.judge.ainvoke(
            {"backbone_output": backbone_output}, run_manager
        )
        return should_continue

    @classmethod
    def create_threshold_evaluator(
        cls: Type["EvaluatorComponent"],
        llm: BaseChatModel,
        threshold: float,
        threshold_mode: str,
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        output_parser: Optional[BaseOutputParser[float]] = None,
    ) -> "EvaluatorComponent[InputType, float]":
        prompt = cls._process_prompt(
            prompt=prompt, user_message=user_message, system_message=system_message
        )

        if output_parser is None:
            output_parser = SimpleEvaluateOutputParser()

        return cls.create_threshold_evaluator_from_runnable(
            runnable=prompt | llm | output_parser,
            threshold=threshold,
            threshold_mode=threshold_mode,
        )

    @classmethod
    def create_threshold_evaluator_from_runnable(
        cls: Type["EvaluatorComponent"],
        runnable: Runnable[InputType, OutputType],
        threshold: float,
        threshold_mode: str,
    ) -> "EvaluatorComponent[InputType, float]":
        if threshold_mode == "leq":
            judge: BaseComponent[Dict[str, float], bool] = LeqThresholdJudge(
                threshold=threshold
            )
        elif threshold_mode == "geq":
            judge = GeqThresholdJudge(threshold=threshold)
        else:
            raise ValueError(
                f"Unknown `threshold_mode` {threshold_mode} when initializing {cls.__name__}."
            )

        backbone = RunnableComponent(runnable)
        return cls(backbone=backbone, judge=judge)
