from abc import ABC, abstractmethod
from textwrap import dedent
from typing import Any, Dict, SupportsFloat, Tuple, Type

import gymnasium as gym
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool


class BaseGameof24Tool(BaseModel, ABC):
    """Base tool for a Game of 24 environment.

    Environment is present as a field, but it won't be shown to models."""

    env: gym.Env = Field(exclude=True)

    class Config(BaseTool.Config):
        pass

    @abstractmethod
    def _operation(self, number1: float, number2: float) -> float: ...

    def _run(
        self,
        number1: int,
        number2: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        reward, terminated, truncated, info = (
            0,
            self.env.is_terminated(),  # type: ignore[attr-defined]
            False,
            {"numbers": self.env.numbers},  # type: ignore[attr-defined]
        )
        if terminated:
            observation = "The environment has already been terminated."
            return observation, reward, terminated, truncated, info

        if not self.env.verify_arguments(number1=number1, number2=number2):  # type: ignore[attr-defined]
            observation = f"Wrong arguments: not all numbers given as arguments to a tool call are available (arguments: {number1}, {number2}, available numbers: {self.env.numbers}."  # type: ignore[attr-defined]
            return observation, reward, terminated, truncated, info

        result = self._operation(number1=number1, number2=number2)

        self.env.remove_number(number1)  # type: ignore[attr-defined]
        self.env.remove_number(number2)  # type: ignore[attr-defined]
        self.env.add_number(result)  # type: ignore[attr-defined]

        observation = f"result of current arithmetical operation on {number1} and {number2} is {result}"
        reward = int(self.env.is_success())  # type: ignore[attr-defined]
        terminated = self.env.is_terminated()  # type: ignore[attr-defined]
        info = {"numbers": self.env.numbers}  # type: ignore[attr-defined]

        return observation, reward, terminated, truncated, info


class CalculatorInput(BaseModel):
    number1: float = Field(description="The first argument in an arithmetical operation.")
    number2: float = Field(description="The second argument in an arithmetical operation.")


class AddTool(BaseGameof24Tool, BaseTool):
    name = "add"
    description = dedent("""
    Adds two numbers. Returns the following:
    * observation: the result of the addition;
    * reward: 1 when the goal is reached (24 is obtained), 0 otherwise;
    * terminated: if True, the game has ended: there's no possible actions anymore;
    * truncated: if True, the time limit has been exceeded;
    * info: the remaining numbers""")
    args_schema: Type[BaseModel] = CalculatorInput  # type: ignore

    def _operation(self, number1: float, number2: float) -> float:
        return number1 + number2


class SubtractTool(BaseGameof24Tool, BaseTool):
    name = "subtract"
    description = dedent("""
    Subtracts the second number from the first one. Returns the following:
    * observation: the result of the subtraction;
    * reward: 1 when the goal is reached (24 is obtained), 0 otherwise;
    * terminated: if True, the game has ended: there's no possible actions anymore;
    * truncated: if True, the time limit has been exceeded;
    * info: the remaining numbers""")
    args_schema: Type[BaseModel] = CalculatorInput  # type: ignore

    def _operation(self, number1: float, number2: float) -> float:
        return number1 - number2


class MultiplyTool(BaseGameof24Tool, BaseTool):
    name = "multiply"
    description = dedent("""
    Multiplies two numbers. Returns the following:
    * observation: the result of the multiplication;
    * reward: 1 when the goal is reached (24 is obtained), 0 otherwise;
    * terminated: if True, the game has ended: there's no possible actions anymore;
    * truncated: if True, the time limit has been exceeded;
    * info: the remaining numbers""")
    args_schema: Type[BaseModel] = CalculatorInput  # type: ignore

    def _operation(self, number1: float, number2: float) -> float:
        return number1 * number2


class DivideTool(BaseGameof24Tool, BaseTool):
    name = "divide"
    description = dedent("""
    Divides the first number by the second one. Returns the following:
    * observation: the result of the division;
    * reward: 1 when the goal is reached (24 is obtained), 0 otherwise;
    * terminated: if True, the game has ended: there's no possible actions anymore;
    * truncated: if True, the time limit has been exceeded;
    * info: the remaining numbers""")
    args_schema: Type[BaseModel] = CalculatorInput  # type: ignore

    def _operation(self, number1: float, number2: float) -> float:
        return number1 / number2
