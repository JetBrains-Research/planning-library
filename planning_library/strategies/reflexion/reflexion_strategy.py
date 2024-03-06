from typing import Any, Dict, Optional, Sequence, Union

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph  # type: ignore[import]
from langgraph.pregel import Pregel  # type: ignore[import-untyped]

from ...utils import convert_runnable_to_agent
from ..base_strategy import BaseLangGraphStrategy
from .components.actors import AgentActor
from .components.evaluators import ReflexionEvaluator
from .components.evaluators.backbones import ReflexionRunnableThoughtEvaluator
from .components.evaluators.continue_judges import ReflexionThresholdEvaluatorContinueJudge
from .components.self_reflections import RunnableSelfReflection
from .reflexion_graph import create_reflexion_graph
from .utils import ReflexionEvaluatorInput


class ReflexionStrategy(BaseLangGraphStrategy):
    @staticmethod
    def create(
        agent: Runnable,
        tools: Sequence[BaseTool],
        evaluator_runnable: Optional[Runnable[ReflexionEvaluatorInput, Any]] = None,
        self_reflection_runnable: Optional[Runnable[Dict[str, Any], Any]] = None,
        max_num_iterations: Optional[int] = None,
        value_threshold: Optional[float] = None,
        **kwargs,
    ) -> Pregel:
        runnable_agent = convert_runnable_to_agent(agent)

        actor = AgentActor(agent=runnable_agent, tools=tools)

        if evaluator_runnable is None:
            raise ValueError("Default evaluator is not provided yet.")

        evaluator = ReflexionEvaluator(
            backbone=ReflexionRunnableThoughtEvaluator(evaluator_runnable),
            judge=ReflexionThresholdEvaluatorContinueJudge(value_threshold if value_threshold else 1.0),
        )

        if self_reflection_runnable is None:
            raise ValueError("Default self reflection runnable is not provided yet.")

        self_reflection = RunnableSelfReflection(self_reflection_runnable)

        return create_reflexion_graph(
            actor=actor, evaluator=evaluator, self_reflection=self_reflection, max_num_iterations=max_num_iterations
        )
