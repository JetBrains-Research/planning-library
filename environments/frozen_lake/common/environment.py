from __future__ import annotations
from typing import Any, Dict, Tuple, Sequence, Optional

import gymnasium as gym
from gymnasium.core import ObsType, SupportsFloat
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from langchain_core.agents import AgentAction
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManager

from .tools import MoveTool
from planning_library.action_executors import DefaultActionExecutor


class FrozenLakeEnvWrapper(gym.Wrapper):
    def __init__(self, env: FrozenLakeEnv):
        super().__init__(env)
        self._action_executor = DefaultActionExecutor(tools=[MoveTool(env=self)])  # type: ignore[call-arg]

    @property
    def tools(self) -> Sequence[BaseTool]:
        return self._action_executor.tools

    def step(
        self, inputs: Tuple[AgentAction, Optional[CallbackManager]]
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        action, run_manager = inputs
        result = self._action_executor.execute(action)
        return result.observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)

        if options is not None and "trajectory" in options:
            for action in options["trajectory"]:
                assert isinstance(action, AgentAction)
                observation, reward, terminated, truncated, info = self.step(
                    (
                        action,
                        options["run_manager"] if "run_manager" in options else None,
                    )
                )
        return observation, info
