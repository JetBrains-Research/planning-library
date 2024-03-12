from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium.core import SupportsFloat
from langchain_core.agents import AgentAction
from langchain_core.tools import BaseTool

from .simple_tools import AddTool, DivideTool, MultiplyTool, SubtractTool


class GameOf24(gym.Env[str, AgentAction]):
    AVAILABLE_ACTIONS: Dict[str, BaseTool] = {
        "add": AddTool(),  # type: ignore[call-arg]
        "multiply": MultiplyTool(),  # type: ignore[call-arg]
        "subtract": SubtractTool(),  # type: ignore[call-arg]
        "divide": DivideTool(),  # type: ignore[call-arg]
    }

    def __init__(self, numbers: Optional[List[int]] = None):
        self.numbers: Dict[float, int] = defaultdict(int)
        if numbers:
            for number in numbers:
                self.numbers[number] += 1

    def __str__(self):
        return " ".join([str(key) for key, value in self.numbers.items() for _ in range(value)])

    def _add_number(self, number: float) -> None:
        self.numbers[number] += 1

    def _remove_number(self, number: float) -> None:
        if number not in self.numbers:
            return

        self.numbers[number] -= 1
        if self.numbers[number] == 0:
            del self.numbers[number]

    def _verify_arguments(self, number1: float, number2: float) -> bool:
        if number1 == number2:
            return number1 in self.numbers and self.numbers[number1] >= 2

        return (
            number1 in self.numbers
            and self.numbers[number1] >= 1
            and number2 in self.numbers
            and self.numbers[number2] >= 1
        )

    def step(self, action: AgentAction) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = None, 0, False, False, {"numbers": str(self)}

        assert isinstance(action.tool_input, dict)
        number1, number2 = float(action.tool_input["number1"]), float(action.tool_input["number2"])

        if not self._verify_arguments(number1=number1, number2=number2):
            observation = "Wrong arguments: not all numbers given as arguments to a tool call are available."
            return observation, reward, terminated, truncated, info

        if action.tool not in self.AVAILABLE_ACTIONS:
            observation = f"Unknown tool. Currently available tools: {list(GameOf24.AVAILABLE_ACTIONS.keys())}."
            return observation, reward, terminated, truncated, info

        result = self.AVAILABLE_ACTIONS[action.tool]._run(number1, number2)

        self._remove_number(number1)
        self._remove_number(number2)
        self._add_number(result)

        observation, info = (
            f"Calling {action.tool} with {number1} and {number2} leads to {result}.",
            {"numbers": str(self)},
        )

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed=seed)

        numbers = options.get("numbers", []) if options else []
        self.numbers = defaultdict(int)
        for number in numbers:
            self.numbers[number] += 1

        if options is None or "trajectory" not in options:
            return "Reset environment.", {"numbers": str(self)}

        trajectory: List[Tuple[AgentAction, str]] = options["trajectory"]
        for action, observation in trajectory:
            self.step(action)
        return "Reset environment.", {"numbers": str(self)}
