from langchain_core.tools import tool


@tool
def add(number1: float, number2: float) -> float:
    """Adds number1 with number2."""
    return number1 + number2


@tool
def subtract(number1: float, number2: float) -> float:
    """Subtracts number2 from number 1."""
    return number1 - number2


@tool
def multiply(number1: float, number2: float) -> float:
    """Multiplies number1 by number2."""
    return number1 * number2


@tool
def divide(number1: float, number2: float) -> float:
    """Divides number1 by number2."""
    return number1 / number2
