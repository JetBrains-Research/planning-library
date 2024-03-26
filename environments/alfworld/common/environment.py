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


class ALFWorldEnv(gym.Env[str, AgentAction]):
    def __init__(
        self,
        config_path: str,
    ):
        with open(config_path) as reader:
            config = yaml.safe_load(reader)

        env = getattr(environment, config["env"]["type"])(config, train_eval="train")
        self.env: TextworldBatchGymEnv = env.init_env(batch_size=1)
        self._action_executor = DefaultActionExecutor(
            tools=get_alfworld_tools(env=self.env)
        )

    @property
    def tools(self) -> Sequence[BaseTool]:
        return self._action_executor.tools

    def seed(self, seed: Optional[int] = None):
        self.env.seed(seed)

    def step(
        self, action: AgentAction
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        result = self._action_executor.execute(action)
        observation, reward, terminated, truncated, info = result.observation
        return observation, reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        obs, infos = self.env.reset()
        observation = obs[0]
        info = {key: infos[key][0] for key in infos}

        if options is not None and "trajectory" in options:
            for action in options["trajectory"]:
                assert isinstance(action, AgentAction)
                observation, reward, terminated, truncated, info = self.step(action)
        return observation, info
