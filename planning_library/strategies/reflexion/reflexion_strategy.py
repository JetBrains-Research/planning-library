from typing import Any, Callable, Dict, Optional, Sequence

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel  # type: ignore[import-untyped]

from ...action_executors import BaseActionExecutor, DefaultActionExecutor
from ...utils import convert_runnable_to_agent
from ..base_strategy import BaseLangGraphStrategy
from .components.actors import AgentActor
from .components.evaluators import ReflexionEvaluator
from .components.evaluators.backbones import ReflexionRunnableThoughtEvaluator
from .components.evaluators.continue_judges import (
    ReflexionThresholdEvaluatorContinueJudge,
)
from .components.self_reflections import RunnableSelfReflection
from .reflexion_graph import create_reflexion_graph
from .utils import ReflexionEvaluatorInput


class ReflexionStrategy(BaseLangGraphStrategy):
    """Reflexion strategy.

    Based on "Reflexion: Language Agents with Verbal Reinforcement Learning", Shinn et al.
    and an example in LangGraph repo: https://github.com/langchain-ai/langgraph/tree/main/examples/reflexion
    """

    @staticmethod
    def create(
        agent: Runnable,
        tools: Sequence[BaseTool],
        action_executor: Optional[BaseActionExecutor] = None,
        evaluator_runnable: Optional[Runnable[ReflexionEvaluatorInput, Any]] = None,
        self_reflection_runnable: Optional[Runnable[Dict[str, Any], Any]] = None,
        max_iterations: Optional[int] = None,
        value_threshold: Optional[float] = None,
        reset_environment: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ) -> Pregel:
        """Creates an instance of Reflexion strategy.

        Reflexion requires evaluator and self-reflection components. The default setting is as follows:
        * evaluator is a runnable that accepts ReflexionEvaluatorInput and returns a float in a 0-1 range;
        * evaluator judges whether to stop execution or to continue to self-reflection based on the threshold;
            execution continues only when the value of the current trial is lower than the given threshold.

        Args:
            agent: The agent to run for proposing thoughts at each DFS step.
            tools: The valid tools the agent can call.
            action_executor: The class responsible for actually executing actions.
            evaluator_runnable: Runnable that powers an evaluator. If None, the default model will be used.
            self_reflection_runnable: Runnable that powers self-reflection. If None, the default model will be used.
            max_iterations: Maximum number of iterations. If None, no restrictions on the number of iterations are imposed.
            value_threshold: Threshold for evaluator; only thoughts evaluated higher than the threshold will be further explored.
            reset_environment: If the agent operates in an environment, this function is responsible for resetting its
             state between Reflexion iterations.
        """
        runnable_agent = convert_runnable_to_agent(agent)

        actor = AgentActor(agent=runnable_agent)

        if evaluator_runnable is None:
            raise ValueError("Default evaluator is not provided yet.")

        evaluator = ReflexionEvaluator(
            backbone=ReflexionRunnableThoughtEvaluator(evaluator_runnable),
            judge=ReflexionThresholdEvaluatorContinueJudge(
                value_threshold if value_threshold else 1.0
            ),
        )

        if self_reflection_runnable is None:
            raise ValueError("Default self reflection runnable is not provided yet.")

        self_reflection = RunnableSelfReflection(self_reflection_runnable)

        if action_executor is None:
            action_executor = DefaultActionExecutor(tools)

        return create_reflexion_graph(
            actor=actor,
            evaluator=evaluator,
            self_reflection=self_reflection,
            action_executor=action_executor,
            max_iterations=max_iterations,
            reset_environment=reset_environment,
        )
