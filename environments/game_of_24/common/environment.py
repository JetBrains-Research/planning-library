from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Sequence

import gymnasium as gym
from gymnasium.core import SupportsFloat
from langchain_core.agents import AgentAction
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManager

from .tools import AddTool, MultiplyTool, SubtractTool, DivideTool
from planning_library.action_executors import DefaultActionExecutor


class GameOf24Env(gym.Env[str, Tuple[AgentAction, Optional[CallbackManager]]]):
    def __init__(self, numbers: Optional[List[float | int]] = None):
        self._action_executor = DefaultActionExecutor(
            tools=[
                AddTool(env=self),  # type: ignore[call-arg]
                MultiplyTool(env=self),  # type: ignore[call-arg]
                SubtractTool(env=self),  # type: ignore[call-arg]
                DivideTool(env=self),  # type: ignore[call-arg]
            ]
        )

        self._numbers: Dict[float, int] = defaultdict(int)
        if numbers:
            for number in numbers:
                self._numbers[float(number)] += 1

    @property
    def numbers(self) -> str:
        return " ".join(
            [str(key) for key, value in self._numbers.items() for _ in range(value)]
        )

    @numbers.setter
    def numbers(self, numbers: List[float | int]):
        self._numbers = defaultdict(int)
        if numbers:
            for number in numbers:
                self._numbers[float(number)] += 1

    @property
    def tools(self) -> Sequence[BaseTool]:
        return self._action_executor.tools

    def is_success(self) -> bool:
        return self._numbers == {24.0: 1}

    def is_terminated(self) -> bool:
        return len(self._numbers) == 1

    def add_number(self, number: float) -> None:
        self._numbers[number] += 1

    def remove_number(self, number: float) -> None:
        if number not in self._numbers:
            return

        self._numbers[number] -= 1
        if self._numbers[number] == 0:
            del self._numbers[number]

    def verify_arguments(self, number1: float, number2: float) -> bool:
        if number1 == number2:
            return number1 in self._numbers and self._numbers[number1] >= 2

        return (
            number1 in self._numbers
            and self._numbers[number1] >= 1
            and number2 in self._numbers
            and self._numbers[number2] >= 1
        )

    def step(
        self, inputs: Tuple[AgentAction, Optional[CallbackManager]]
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        action, run_manager = inputs
        result = self._action_executor.execute(action, run_manager=run_manager)
        return result.observation

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed=seed)

        self.numbers = options.get("numbers", []) if options else []

        observation, info = "", {"numbers": self.numbers}

        if options is not None and "trajectory" in options:
            for action, step in options["trajectory"]:
                assert isinstance(
                    action, AgentAction
                ), f"Expected AgentAction, got {action}"
                observation, reward, terminated, truncated, info = self.step(
                    (
                        action,
                        options["run_manager"] if "run_manager" in options else None,
                    )
                )

        return observation, info
