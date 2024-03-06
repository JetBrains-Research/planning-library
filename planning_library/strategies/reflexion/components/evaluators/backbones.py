from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager
from langchain_core.runnables import Runnable

from ...utils import BaseEvaluatorInput, ReflexionEvaluatorInput


class BaseEvaluatorBackbone(ABC):
    """A base evaluator backbone. It is responsible for actually evaluating each proposed thought."""

    @abstractmethod
    def evaluate(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> Any:
        ...

    @abstractmethod
    async def aevaluate(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> Any:
        ...


class RunnableEvaluator(BaseEvaluatorBackbone):
    """A thought evaluator backbone powered by a Runnable."""

    def __init__(self, runnable: Runnable[BaseEvaluatorInput, Any]):
        self.runnable = runnable

    def evaluate(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> Any:
        # TODO: clean up the mess with evaluators
        return self.runnable.invoke(
            {"inputs": inputs, "intermediate_steps": intermediate_steps, **kwargs},  # type: ignore[typeddict-item]
            {"callbacks": run_manager} if run_manager else {},
        )

    async def aevaluate(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> Any:
        # TODO: clean up the mess with evaluators
        result = await self.runnable.ainvoke(
            {"inputs": inputs, "intermediate_steps": intermediate_steps, **kwargs},  # type: ignore[typeddict-item]
            {"callbacks": run_manager} if run_manager else {},
        )
        return result


class ReflexionBaseEvaluatorBackbone(ABC):
    @abstractmethod
    def evaluate(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        agent_outcome: Union[List[AgentAction], AgentAction, AgentFinish],
        run_manager: Optional[CallbackManager] = None,
    ) -> Any:
        ...

    @abstractmethod
    async def aevaluate(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        agent_outcome: Union[List[AgentAction], AgentAction, AgentFinish],
        run_manager: Optional[CallbackManager] = None,
    ) -> Any:
        ...


class ReflexionRunnableThoughtEvaluator(ReflexionBaseEvaluatorBackbone):
    """A reflexion evaluator backbone powered by a Runnable."""

    def __init__(self, runnable: Runnable[ReflexionEvaluatorInput, Any]):
        # TODO: clean up the mess with evaluators
        self.evaluator = RunnableEvaluator(runnable)  # type: ignore[arg-type]

    def evaluate(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        agent_outcome: Union[List[AgentAction], AgentAction, AgentFinish],
        run_manager: Optional[CallbackManager] = None,
    ) -> Any:
        return self.evaluator.evaluate(
            inputs=inputs, intermediate_steps=intermediate_steps, agent_outcome=agent_outcome, run_manager=run_manager
        )

    async def aevaluate(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        agent_outcome: Union[List[AgentAction], AgentAction, AgentFinish],
        run_manager: Optional[CallbackManager] = None,
    ) -> Any:
        return await self.evaluator.aevaluate(
            inputs=inputs, intermediate_steps=intermediate_steps, agent_outcome=agent_outcome, run_manager=run_manager
        )
