import asyncio
from typing import Any, Dict, List, Optional, overload

from langchain_core.agents import AgentAction, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.tools import BaseTool

from ..utils.actions_utils import aperform_agent_action, perform_agent_action
from .base_action_executor import BaseActionExecutor


class DefaultActionExecutor(BaseActionExecutor):
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

    def execute(
        self,
        actions: List[AgentAction] | AgentAction,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        verbose: bool = True,
        tool_run_logging_kwargs: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        **kwargs,
    ) -> List[AgentStep] | AgentStep:
        if isinstance(actions, AgentAction):
            tool_result = perform_agent_action(
                agent_action=actions,
                name_to_tool_map=name_to_tool_map,
                color_mapping=color_mapping,
                verbose=verbose,
                tool_run_kwargs=tool_run_logging_kwargs,
                run_manager=run_manager,
            )
            return tool_result
        elif isinstance(actions, list):
            observations = []
            for action in actions:
                tool_result = perform_agent_action(
                    agent_action=action,
                    name_to_tool_map=name_to_tool_map,
                    color_mapping=color_mapping,
                    verbose=verbose,
                    tool_run_kwargs=tool_run_logging_kwargs,
                    run_manager=run_manager,
                )
                observations.append(tool_result)
            return observations

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
        if isinstance(actions, AgentAction):
            tool_result = await aperform_agent_action(
                agent_action=actions,
                name_to_tool_map=name_to_tool_map,
                color_mapping=color_mapping,
                verbose=verbose,
                tool_run_kwargs=tool_run_logging_kwargs,
                run_manager=run_manager,
            )
            return tool_result
        elif isinstance(actions, list):
            # TODO: no idea why mypy complains
            with asyncio.TaskGroup() as tg:  # type: ignore[attr-defined]
                tool_results = [
                    tg.create_task(
                        aperform_agent_action(
                            agent_action=action,
                            name_to_tool_map=name_to_tool_map,
                            color_mapping=color_mapping,
                            verbose=verbose,
                            tool_run_kwargs=tool_run_logging_kwargs,
                            run_manager=run_manager,
                        )
                    )
                    for action in actions
                ]
            return [task.result() for task in tool_results]
