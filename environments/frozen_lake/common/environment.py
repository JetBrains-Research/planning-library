from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, SupportsFloat, Tuple

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from langchain_core.agents import AgentAction
from langchain_core.callbacks import CallbackManager
from langchain_core.tools import BaseTool

from planning_library.action_executors import LangchainActionExecutor

from .tools import MoveTool


class FrozenLakeEnvWrapper(gym.Wrapper):
    def __init__(self, env: FrozenLakeEnv):
        super().__init__(env)
        self._action_executor = LangchainActionExecutor(tools=[MoveTool(env=self)])  # type: ignore[call-arg]

    @property
    def tools(self) -> Sequence[BaseTool]:
        return self._action_executor.tools

    def step(
        self, action: Tuple[AgentAction, Optional[CallbackManager]]
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        lc_action, run_manager = action
        result = self._action_executor.execute(lc_action, run_manager=run_manager)
        return result.observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
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
