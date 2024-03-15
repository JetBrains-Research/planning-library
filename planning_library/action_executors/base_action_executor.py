from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, overload

from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.tools import BaseTool


class BaseActionExecutor(ABC):
    @overload
    def execute(
        self,
        actions: List[AgentAction],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        verbose: bool = True,
        tool_run_logging_kwargs: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    def execute(
        self,
        actions: AgentAction,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        verbose: bool = True,
        tool_run_logging_kwargs: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        **kwargs,
    ) -> AgentStep: ...

    @abstractmethod
    def execute(
        self,
        actions: List[AgentAction] | AgentAction | AgentFinish,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        verbose: bool = True,
        tool_run_logging_kwargs: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        **kwargs,
    ) -> List[AgentStep] | AgentStep:
        """Performs actions.

        Args:
            actions: Currently proposed actions. Can be: multi-action, single action, finishing.
            name_to_tool_map: Mapping from tool names to actual tools, used for calling tools based on agent's output.
            color_mapping: Mapping from tool names to colors, used for logging purposes when calling tools.
            run_manager: Callback for the current run.

        Returns:
              * List[AgentStep] - for multi-action thoughts (List[AgentAction])
              * AgentStep - for single-action thoughts (AgentAction)
              * None - for finishing thoughts (AgentFinish)
        """
        ...

    @overload
    async def aexecute(
        self,
        actions: List[AgentAction],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        verbose: bool = True,
        tool_run_logging_kwargs: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
        **kwargs,
    ) -> List[AgentStep]: ...

    @overload
    async def aexecute(
        self,
        actions: AgentAction,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        verbose: bool = True,
        tool_run_logging_kwargs: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
        **kwargs,
    ) -> AgentStep: ...

    @abstractmethod
    async def aexecute(
        self,
        actions: List[AgentAction] | AgentAction,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        verbose: bool = True,
        tool_run_logging_kwargs: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
        **kwargs,
    ) -> List[AgentStep] | AgentStep:
        """Performs actions asynchronously.

        Args:
            actions: Currently proposed actions. Can be: multi-action, single action, finishing.
            name_to_tool_map: Mapping from tool names to actual tools, used for calling tools based on agent's output.
            color_mapping: Mapping from tool names to colors, used for logging purposes when calling tools.
            run_manager: Callback for the current run.

        Returns:
              * List[AgentStep] - for multi-action thoughts (List[AgentAction])
              * AgentStep - for single-action thoughts (AgentAction)
              * None - for finishing thoughts (AgentFinish)
        """
        ...
