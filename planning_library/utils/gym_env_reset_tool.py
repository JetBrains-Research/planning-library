from __future__ import annotations

from typing import Any, Dict, Generic, Optional, Tuple, TypeVar

import gymnasium as gym
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.agents import AgentAction
from langchain_core.callbacks import CallbackManager

ObsType = TypeVar("ObsType")


class GymEnvResetTool(BaseTool, BaseModel, Generic[ObsType]):
    env: gym.Env[ObsType, Tuple[AgentAction, Optional[CallbackManager]]] = Field(  # type: ignore[valid-type]
        exclude=True
    )

    name: str = "reset"
    description: str = "Resets the environment state."

    class Config(BaseTool.Config):
        pass

    def _run(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)  # type: ignore[reportReturnType]
