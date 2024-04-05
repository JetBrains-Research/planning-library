from typing import Optional, Dict, Generic, Type

from langchain_core.callbacks import CallbackManager, AsyncCallbackManager
from langchain_core.output_parsers import BaseOutputParser
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

        prompt = cls._process_prompt(
            prompt=prompt, user_message=user_message, system_message=system_message
        )

        if output_parser is None:
            output_parser = SimpleEvaluateOutputParser()

        backbone = RunnableComponent(runnable=prompt | llm | output_parser)
        return cls(backbone=backbone, judge=judge)
