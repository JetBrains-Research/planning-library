from abc import ABC, abstractmethod
from typing import List, overload, Sequence

from langchain_core.agents import AgentAction, AgentStep
from langchain_core.tools import BaseTool


class BaseActionExecutor(ABC):
    @property
    @abstractmethod
    def tools(self) -> Sequence[BaseTool]: ...

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
            actions: Currently proposed actions. Can be: multi-action, single action, finishing.
            run_manager: Callback for the current run.

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
            actions: Currently proposed actions. Can be: multi-action, single action, finishing.

        Returns:
              * List[AgentStep] - for multi-action thoughts (List[AgentAction])
              * AgentStep - for single-action thoughts (AgentAction)
        """
        ...
