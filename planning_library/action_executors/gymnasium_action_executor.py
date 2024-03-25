from typing import List, Optional, overload, Sequence

import gymnasium as gym
from gymnasium.core import ObsType
from langchain_core.agents import AgentAction, AgentStep

from .base_action_executor import BaseActionExecutor
from langchain_core.tools import BaseTool


class GymnasiumActionExecutor(BaseActionExecutor):
    def __init__(
        self,
        env: gym.Env[
            ObsType,
            AgentAction,
        ],
        seed: Optional[int] = None,
    ):
        self._env = env
        self._seed = seed

    @property
    def tools(self) -> Sequence[BaseTool]:
        return self._env.get_wrapper_attr("tools")

    def reset(self, actions: Optional[List[AgentAction]] = None, **kwargs) -> None:
        """Resets the environment. If actions are passed, will also execute them."""

        options = kwargs
        if actions:
            options["actions"] = actions

        self._env.reset(seed=self._seed, options=options)

    @overload
    def execute(
        self,
        actions: List[AgentAction],
        reset_env_before_action: bool = False,
        **reset_kwargs,
    ) -> List[AgentStep]: ...

    @overload
    def execute(
        self,
        actions: AgentAction,
        reset_env_before_action: bool = False,
        **reset_kwargs,
    ) -> AgentStep: ...

    def execute(
        self,
        actions: List[AgentAction] | AgentAction,
        reset_env_before_action: bool = False,
        **reset_kwargs,
    ) -> List[AgentStep] | AgentStep:
        if reset_env_before_action:
            self.reset(**reset_kwargs)

        if isinstance(actions, AgentAction):
            observation, reward, terminated, truncated, info = self._env.step(actions)

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
                reset_env_before_action=reset_env_before_action,
                **reset_kwargs,
            )
            for action in actions
        ]

    @overload
    async def aexecute(
        self,
        actions: List[AgentAction],
        reset_before_action: bool = False,
        **reset_kwargs,
    ) -> List[AgentStep]: ...

    @overload
    async def aexecute(
        self,
        actions: AgentAction,
        reset_before_action: bool = False,
        **reset_kwargs,
    ) -> AgentStep: ...

    async def aexecute(
        self,
        actions: List[AgentAction] | AgentAction,
        reset_before_action: bool = False,
        **reset_kwargs,
    ) -> List[AgentStep] | AgentStep:
        raise NotImplementedError()
