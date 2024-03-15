import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterator, Optional

from langchain_core.callbacks import Callbacks
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping

from .base_strategy import BaseCustomStrategy

logger = logging.getLogger(__name__)


class BaseCustomStrategyIterator(ABC):
    """Base iterator for a strategy.

    Heavily based on langchain.agents.agent_iterator.AgentExecutorIterator.
    """

    def __init__(
        self,
        strategy: BaseCustomStrategy,
        inputs: Any,
        callbacks: Callbacks = None,
        *,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        include_run_info: bool = False,
        yield_actions: bool = False,
    ):
        """
        Initialize the iterator with the given strategy, inputs, and optional callbacks.
        """
        self._strategy = strategy
        self.inputs = inputs
        self.callbacks = callbacks
        self.tags = tags
        self.metadata = metadata
        self.run_name = run_name
        self.include_run_info = include_run_info
        self.yield_actions = yield_actions
        self.reset()

    _inputs: Dict[str, str]
    callbacks: Callbacks
    tags: Optional[list[str]]
    metadata: Optional[Dict[str, Any]]
    run_name: Optional[str]
    include_run_info: bool
    yield_actions: bool

    @property
    def inputs(self) -> Dict[str, str]:
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: Any) -> None:
        self._inputs = self.strategy.prep_inputs(inputs)

    @property
    def strategy(self) -> BaseCustomStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: BaseCustomStrategy) -> None:
        self._strategy = strategy
        # force re-prep inputs in case agent_executor's prep_inputs fn changed
        self.inputs = self.inputs

    @property
    def name_to_tool_map(self) -> Dict[str, BaseTool]:
        return {tool.name: tool for tool in self.strategy.tools}

    @property
    def color_mapping(self) -> Dict[str, str]:
        return get_color_mapping(
            [tool.name for tool in self.strategy.tools],
            excluded_colors=["green", "red"],
        )

    def reset(self) -> None:
        """
        Reset the iterator to its initial state, clearing intermediate steps,
        iterations, and time elapsed.
        """
        logger.debug("(Re)setting StrategyIterator to fresh state")
        self.iterations = 0
        # maybe better to start these on the first __anext__ call?
        self.time_elapsed = 0.0
        self.start_time = time.time()

    def update_iterations(self) -> None:
        """
        Increment the number of iterations and update the time elapsed.
        """
        self.iterations += 1
        self.time_elapsed = time.time() - self.start_time
        logger.debug(
            f"Agent Iterations: {self.iterations} ({self.time_elapsed:.2f}s elapsed)"
        )

    @abstractmethod
    def __iter__(self: "BaseCustomStrategyIterator") -> Iterator[AddableDict]: ...

    @abstractmethod
    async def __aiter__(
        self: "BaseCustomStrategyIterator",
    ) -> AsyncIterator[AddableDict]: ...
