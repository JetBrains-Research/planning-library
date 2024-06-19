from typing import Any, Callable, Dict, Optional

from ...action_executors import BaseActionExecutor
from ..base_strategy import BaseLangGraphStrategy
from .components import ReflexionActor, ReflexionEvaluator, ReflexionSelfReflection
from .reflexion_graph import create_reflexion_graph


class ReflexionStrategy(BaseLangGraphStrategy):
    """Reflexion strategy.

    Based on "Reflexion: Language Agents with Verbal Reinforcement Learning", Shinn et al.
    and an example in LangGraph repo: https://github.com/langchain-ai/langgraph/tree/main/examples/reflexion
    """

    @staticmethod
    def create_from_components(
        actor: ReflexionActor,
        evaluator: ReflexionEvaluator,
        self_reflection: ReflexionSelfReflection,
        action_executor: BaseActionExecutor,
        max_iterations: int,
        reset_environment: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        return create_reflexion_graph(
            actor=actor,
            evaluator=evaluator,
            self_reflection=self_reflection,
            action_executor=action_executor,
            max_iterations=max_iterations,
            reset_environment=reset_environment,
        )
