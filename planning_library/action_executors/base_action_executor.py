from abc import ABC, abstractmethod
from typing import List, overload, Sequence, Optional

from langchain_core.agents import AgentAction, AgentStep
from langchain_core.tools import BaseTool


class BaseActionExecutor(ABC):
    @property
    @abstractmethod
    def tools(self) -> Sequence[BaseTool]: ...

    @abstractmethod
    def reset(self, actions: Optional[List[AgentAction]] = None, **kwargs) -> None:
        """Resets the current state. If actions are passed, will also execute them."""
        ...

    @overload
    def execute(
        self,
        actions: List[AgentAction],
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    def execute(
        self,
        actions: AgentAction,
        **kwargs,
    ) -> AgentStep: ...

    @abstractmethod
    def execute(
        self,
        actions: List[AgentAction] | AgentAction,
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
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    async def aexecute(
        self,
        actions: AgentAction,
        **kwargs,
    ) -> AgentStep: ...

    @abstractmethod
    async def aexecute(
        self,
        actions: List[AgentAction] | AgentAction,
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
