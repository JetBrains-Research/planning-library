from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional, Type, Union

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool


class GameOf24Environment:
    """Basic Game of 24 environment."""

    def __init__(self, numbers: Optional[Counter[float]] = None, numbers_str: Optional[str] = None):
        self.numbers: Optional[Counter[float]] = None
        if numbers_str is not None:
            self.numbers = Counter(float(n) for n in numbers_str.split())
        elif numbers is not None:
            self.numbers = numbers

    def is_present(self, number: float, quantity: int = 1) -> bool:
        assert self.numbers is not None, "Numbers are not set!"
        return self.numbers[number] >= quantity

    def update(self, *args):
        self.numbers = Counter(n for n in args)

    def __repr__(self):
        return " ".join([str(key) for key, value in self.numbers.most_common() for _ in range(value)])


class BaseGameOf24Tool(BaseModel):
    """Base tool for a Game of 24 environment.

    Environment is present as a field, but it won't be shown to models."""

    env: GameOf24Environment = Field(exclude=True)

    class Config(BaseTool.Config):
        pass


class GetRemainingNumbersInput(BaseModel):
    pass


class GetRemainingNumbersGameOf24Tool(BaseGameOf24Tool, BaseTool):
    """Outputs a space separated list of all remaining numbers."""

    name: str = "get_remaining_numbers"
    description: str = "Outputs a space separated list of all remaining numbers."
    args_schema: Type[BaseModel] = GetRemainingNumbersInput

    def _run(
        self,
        tool_input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        return str(self.env)


class GameOf24OperationInput(BaseModel):
    number1: float = Field(description="First number in an arithmetic operation.")
    number2: float = Field(description="Second number in an arithmetic operation.")


class BaseGameOf24OperationTool(BaseGameOf24Tool, ABC):
    """Base tool for an arithmetic operation in a Game of 24 environment."""

    args_schema: Optional[Type[BaseModel]] = GameOf24OperationInput

    @abstractmethod
    def _operation(self, number1: float, number2: float) -> float:
        ...

    def _run(
        self,
        number1: float,
        number2: float,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Union[GameOf24Environment, str]:
        if number1 == number2 and not self.env.is_present(number1, quantity=2):
            return f"{number1} and {number2} are not present in currently available numbers."
        if not self.env.is_present(number1):
            return f"{number1} is not present in currently available numbers."
        if not self.env.is_present(number2):
            return f"{number2} is not present in currently available numbers."

        result = self._operation(number1=number1, number2=number2)
        assert self.env.numbers is not None, "Numbers are not set!"
        new_numbers = self.env.numbers.copy()
        new_numbers.update({number1: -1, number2: -1, result: 1})
        new_numbers = Counter(number for number, count in new_numbers.most_common() if count > 0)
        return GameOf24Environment(numbers=new_numbers)


class AddGameOf24Tool(BaseGameOf24OperationTool, BaseTool):
    """Add two numbers."""

    name: str = "add"
    description: str = """
        Add two numbers.
        If any of the numbers are not present in currently available numbers, an error message will be returned.
        """

    def _operation(self, number1: float, number2: float) -> float:
        return number1 + number2


class SubtractGameOf24Tool(BaseGameOf24OperationTool, BaseTool):
    """Subtract two numbers (number2 is subtracted from number1)."""

    name: str = "subtract"
    description: str = """
        Subtract two numbers (number2 is subtracted from number1).
        If any of the numbers are not present in currently available numbers, an error message will be returned.
        """

    def _operation(self, number1: float, number2: float) -> float:
        return number1 - number2


class MultiplyGameOf24Tool(BaseGameOf24OperationTool, BaseTool):
    """Multiply two numbers."""

    name: str = "multiply"
    description: str = """
        Multiply two numbers.
        If any of the numbers are not present in currently available numbers, an error message will be returned.
        """

    def _operation(self, number1: float, number2: float) -> float:
        return number1 * number2


class DivideGameOf24Tool(BaseGameOf24OperationTool, BaseTool):
    """Divide two numbers (number1 is divided by number2)."""

    name: str = "divide"
    description: str = """
        Divide two numbers (number1 is divided by number2).
        If any of the numbers are not present in currently available numbers, an error message will be returned.
        """

    def _operation(self, number1: float, number2: float) -> float:
        return number1 / number2
