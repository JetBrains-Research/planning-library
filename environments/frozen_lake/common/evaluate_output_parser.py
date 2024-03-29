import re

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser


class FrozenMapEvaluateOutputParser(BaseOutputParser[float]):
    def parse(self, text: str) -> float:
        try:
            match = re.search(r"\[\[(.*?)\]\]", text.strip())
            if not match:
                raise ValueError("Pattern [[number]] not found.")
            result = float(match.groups()[0])
            if result < 0.0 or result > 1.0:
                raise ValueError("The given number is out of (0.0, 1.0) range.")
            return result
        except ValueError:
            raise OutputParserException(
                f"Couldn't convert {text} to float between 0 and 1."
            )
