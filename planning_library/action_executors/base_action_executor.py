from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, overload, Sequence, Optional
from langchain_core.agents import AgentAction, AgentStep
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManager, AsyncCallbackManager


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

    @overload
    def execute(
        self,
        actions: List[AgentAction],
        run_manager: Optional[CallbackManager] = None,
        reset_before_action: bool = False,
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    def execute(
        self,
        actions: AgentAction,
        run_manager: Optional[CallbackManager] = None,
        reset_before_action: bool = False,
        **kwargs,
    ) -> AgentStep: ...

    @abstractmethod
    def execute(
        self,
        actions: List[AgentAction] | AgentAction,
        run_manager: Optional[CallbackManager] = None,
        reset_before_action: bool = False,
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
    async def aexecute(
        self,
        actions: List[AgentAction],
        run_manager: Optional[AsyncCallbackManager] = None,
        reset_before_action: bool = False,
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    async def aexecute(
        self,
        actions: AgentAction,
        run_manager: Optional[AsyncCallbackManager] = None,
        reset_before_action: bool = False,
        **kwargs,
    ) -> AgentStep: ...

    @abstractmethod
    async def aexecute(
        self,
        actions: List[AgentAction] | AgentAction,
        run_manager: Optional[AsyncCallbackManager] = None,
        reset_before_action: bool = False,
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
