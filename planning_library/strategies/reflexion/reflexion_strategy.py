from typing import Any, Callable, Dict, Optional, Sequence

from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel  # type: ignore[import-untyped]

from ...action_executors import BaseActionExecutor, DefaultActionExecutor
from ..base_strategy import BaseLangGraphStrategy
from .reflexion_graph import create_reflexion_graph
from .components import ReflexionActor, ReflexionSelfReflection, ReflexionEvaluator


class ReflexionStrategy(BaseLangGraphStrategy):
    """Reflexion strategy.

    Based on "Reflexion: Language Agents with Verbal Reinforcement Learning", Shinn et al.
    and an example in LangGraph repo: https://github.com/langchain-ai/langgraph/tree/main/examples/reflexion
    """

    @staticmethod
    def create(
        tools: Sequence[BaseTool],
        action_executor: Optional[BaseActionExecutor] = None,
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
            tools: The valid tools the agent can call.
            action_executor: The class responsible for actually executing actions.
            max_iterations: Maximum number of iterations. If None, no restrictions on the number of iterations are imposed.
            value_threshold: Threshold for evaluator; only thoughts evaluated higher than the threshold will be further explored.
            reset_environment: If the agent operates in an environment, this function is responsible for resetting its
             state between Reflexion iterations.
        """
        actor_kwargs = {
            key[len("actor_") :]: kwargs[key]
            for key in kwargs
            if key.startswith("actor_")
        }
        if "runnable" in actor_kwargs:
            actor = ReflexionActor(agent=actor_kwargs["runnable"])
        elif "agent" in actor_kwargs:
            actor = ReflexionActor(agent=actor_kwargs["agent"])
        else:
            actor = ReflexionActor.create(tools=tools, **actor_kwargs)

        evaluator_kwargs = {
            key[len("evaluator_") :]: kwargs[key]
            for key in kwargs
            if key.startswith("evaluator_")
        }
        if "runnable" in evaluator_kwargs:
            evaluator = ReflexionEvaluator.create_from_runnable(
                runnable=evaluator_kwargs["runnable"], threshold=value_threshold
            )
        else:
            evaluator = ReflexionEvaluator.create(
                threshold=value_threshold, **evaluator_kwargs
            )

        self_reflection_kwargs = {
            key[len("self_reflection_") :]: kwargs[key]
            for key in kwargs
            if key.startswith("self_reflection_")
        }
        if "runnable" in self_reflection_kwargs:
            self_reflection = ReflexionSelfReflection(
                runnable=self_reflection_kwargs["runnable"]
            )
        else:
            self_reflection = ReflexionSelfReflection.create(**self_reflection_kwargs)

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
