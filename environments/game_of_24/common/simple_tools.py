from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool


class CalculatorInput(BaseModel):
    number1: float = Field(description="first number")
    number2: float = Field(description="second number")


class AddTool(BaseTool):
    name = "add"
    description = "Adds two numbers."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, number1: float, number2: float, run_manager: Optional[CallbackManagerForToolRun] = None) -> float:
        """Use the tool."""
        return number1 + number2


class SubtractTool(BaseTool):
    name = "subtract"
    description = "Subtracts number1 from number2."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, number1: float, number2: float, run_manager: Optional[CallbackManagerForToolRun] = None) -> float:
        return number1 - number2


class MultiplyTool(BaseTool):
    name = "multiply"
    description = "Multiplies two numbers."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, number1: float, number2: float, run_manager: Optional[CallbackManagerForToolRun] = None) -> float:
        return number1 * number2


class DivideTool(BaseTool):
    name = "divide"
    description = "Divides number1 by number2."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, number1: float, number2: float, run_manager: Optional[CallbackManagerForToolRun] = None) -> float:
        return number1 / number2
