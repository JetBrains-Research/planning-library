from typing import Any, Dict, List, Optional, Tuple, overload

import gymnasium as gym
from gymnasium.core import ObsType
from langchain_core.agents import AgentAction, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.tools import BaseTool

from .base_action_executor import BaseActionExecutor


class GymnasiumActionExecutor(BaseActionExecutor):
    def __init__(
        self,
        env: gym.Env[
            ObsType,
            Tuple[AgentAction, Optional[CallbackManagerForChainRun], Dict[str, Any]],
        ],
        seed: Optional[int] = None,
    ):
        self._env = env
        self._seed = seed

    @overload
    def execute(
        self,
        actions: List[AgentAction],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        verbose: bool = True,
        tool_run_logging_kwargs: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        reset_env_before_action: bool = False,
        **reset_kwargs,
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
        reset_env_before_action: bool = False,
        **reset_kwargs,
    ) -> AgentStep: ...

    def execute(
        self,
        actions: List[AgentAction] | AgentAction,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        verbose: bool = True,
        tool_run_logging_kwargs: Optional[Dict[str, Any]] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
        reset_env_before_action: bool = False,
        **reset_kwargs,
    ) -> List[AgentStep] | AgentStep:
        tool_run_logging_kwargs = (
            {} if tool_run_logging_kwargs is None else tool_run_logging_kwargs
        )

        if reset_env_before_action:
            self._env.reset(seed=self._seed, options=reset_kwargs)

        if isinstance(actions, AgentAction):
            observation, reward, terminated, truncated, info = self._env.step(
                (actions, run_manager, tool_run_logging_kwargs)
            )

            return AgentStep(
                action=actions,
                observation={
                    "observation": observation,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                },
            )
        return [
            self.execute(
                actions=action,
                name_to_tool_map=name_to_tool_map,
                color_mapping=color_mapping,
                verbose=verbose,
                tool_run_logging_kwargs=tool_run_logging_kwargs,
                run_manager=run_manager,
                reset_env_before_action=reset_env_before_action,
                **reset_kwargs,
            )
            for action in actions
        ]

    @overload
    async def aexecute(
        self,
        actions: List[AgentAction],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        verbose: bool = True,
        tool_run_logging_kwargs: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
        reset_before_action: bool = False,
        **reset_kwargs,
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
        reset_before_action: bool = False,
        **reset_kwargs,
    ) -> AgentStep: ...

    async def aexecute(
        self,
        actions: List[AgentAction] | AgentAction,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        verbose: bool = True,
        tool_run_logging_kwargs: Optional[Dict[str, Any]] = None,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
        reset_before_action: bool = False,
        **reset_kwargs,
    ) -> List[AgentStep] | AgentStep:
        raise NotImplementedError()
