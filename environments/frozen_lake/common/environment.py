from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium.core import ObsType, SupportsFloat
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from langchain_core.agents import AgentAction
from langchain_core.callbacks import CallbackManagerForChainRun

from .tools import (
    MoveTool,
)
from planning_library.utils import get_tools_maps, perform_agent_action


class FrozenLakeEnvWrapper(gym.Wrapper):
    def __init__(self, env: FrozenLakeEnv):
        super().__init__(env)
        self.name_to_tool_map, self.color_mapping = get_tools_maps(
            [
                MoveTool(env=self.env.unwrapped),  # type: ignore[call-arg]
            ]
        )
        # CheckPositionTool(env=self.env.unwrapped),])
        # CheckMapTool(env=self.env.unwrapped)])

    def step(
        self,
        cur_input: Tuple[
            AgentAction, Optional[CallbackManagerForChainRun], Dict[str, Any]
        ],
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        action, run_manager, tool_run_logging_kwargs = cur_input
        result = perform_agent_action(
            agent_action=action,
            name_to_tool_map=self.name_to_tool_map,
            color_mapping=self.color_mapping,
            run_manager=run_manager,
            tool_run_kwargs=tool_run_logging_kwargs,
        )
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
                    (action, None, {})
                )
        return observation, info
