from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, overload

from langchain_core.agents import AgentAction, AgentStep
from langchain_core.callbacks import AsyncCallbackManager, CallbackManager
from langchain_core.tools import BaseTool


class BaseActionExecutor(ABC):
    @property
    @abstractmethod
    def tools(self) -> Sequence[BaseTool]: ...

    @abstractmethod
    def reset(
        self,
        actions: Optional[List[AgentAction]] = None,
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> None:
        """Resets the current state. If actions are passed, will also execute them."""
        ...

    @abstractmethod
    async def areset(
        self,
        actions: Optional[List[AgentAction]] = None,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> None:
        """Resets the current state. If actions are passed, will also execute them."""
        ...

    @overload
    @abstractmethod
    def execute(
        self,
        actions: List[AgentAction],
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    @abstractmethod
    def execute(
        self,
        actions: AgentAction,
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> AgentStep: ...

    @abstractmethod
    def execute(
        self,
        actions: List[AgentAction] | AgentAction,
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> List[AgentStep] | AgentStep:
        """Performs actions.

        Args:
            actions: Currently proposed actions. Can be: multi-action, single action.

        Returns:
              * List[AgentStep] - for multi-action thoughts (List[AgentAction])
              * AgentStep - for single-action thoughts (AgentAction)
        """
        ...

    @overload
    @abstractmethod
    async def aexecute(
        self,
        actions: List[AgentAction],
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    @abstractmethod
    async def aexecute(
        self,
        actions: AgentAction,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> AgentStep: ...

    @abstractmethod
    async def aexecute(
        self,
        actions: List[AgentAction] | AgentAction,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> List[AgentStep] | AgentStep:
        """Performs actions asynchronously.

        Args:
            actions: Currently proposed actions. Can be: multi-action, single action.

        Returns:
              * List[AgentStep] - for multi-action thoughts (List[AgentAction])
              * AgentStep - for single-action thoughts (AgentAction)
        """
        ...
