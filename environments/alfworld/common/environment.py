from __future__ import annotations
import gymnasium as gym
from langchain_core.agents import AgentAction
from langchain_core.tools import BaseTool
import alfworld.agents.environment as environment  # type: ignore[import-untyped]
import yaml  # type: ignore[import-untyped]
from typing import Dict, Any, Tuple, Optional, Sequence
from gymnasium.core import SupportsFloat
from .tools import get_alfworld_tools
from planning_library.action_executors import DefaultActionExecutor
from textworld.gym.envs.textworld_batch import TextworldBatchGymEnv  # type: ignore[import-untyped]
from langchain_core.callbacks import CallbackManager

from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv  # type: ignore[import-untyped]


class ALFWorldEnv(gym.Env[str, Tuple[AgentAction, Optional[CallbackManager]]]):
    def __init__(
        self,
        config_path: str,
    ):
        with open(config_path) as reader:
            config = yaml.safe_load(reader)
        self._alfworld_env: AlfredTWEnv = getattr(environment, config["env"]["type"])(
            config, train_eval="train"
        )
        self.env: TextworldBatchGymEnv = self._alfworld_env.init_env(batch_size=1)
        self._action_executor = DefaultActionExecutor(
            tools=get_alfworld_tools(env=self.env)
        )

    @property
    def tools(self) -> Sequence[BaseTool]:
        return self._action_executor.tools

    def seed(self, seed: Optional[int] = None):
        self.env.seed(seed)

    def step(
        self, inputs: Tuple[AgentAction, Optional[CallbackManager]]
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        action, run_manager = inputs
        result = self._action_executor.execute(action, run_manager=run_manager)
        try:
            observation, reward, terminated, truncated, info = result.observation
        except ValueError:
            observation = result.observation
            reward = 0
            terminated = False
            truncated = False

        return observation, reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        if not options or "next_episode" not in options or not options["next_episode"]:
            self.env = self._alfworld_env.init_env(batch_size=1)
            self._action_executor = DefaultActionExecutor(
                tools=get_alfworld_tools(env=self.env)
            )

        obs, infos = self.env.reset()
        observation = obs[0]
        info = {key: infos[key][0] for key in infos}

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
