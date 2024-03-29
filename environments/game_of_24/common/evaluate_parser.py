from typing import Dict, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser


class GameOf24EvaluateOutputParser(BaseOutputParser[float]):
    values_map: Dict[str, float] = {"sure": 1.0, "likely": 0.5, "impossible": 0.0}
    parser_mode: Literal["strict", "loose"] = "loose"

    def _parse_strict(self, cleaned_text: str) -> float:
        """Expects text to BE one of the allowed values."""
        if cleaned_text not in self.values_map:
            raise OutputParserException(
                f"{self.__class__.__name__} expected output value to be "
                f"one of {self.values_map.keys()} (case-insensitive). "
                f"Received {cleaned_text}."
            )
        return self.values_map[cleaned_text]

    def _parse_loose(self, cleaned_text: str) -> float:
        """Expects text to CONTAIN one of the allowed values."""
        for value in self.values_map:
            if value in cleaned_text:
                return self.values_map[value]

        raise OutputParserException(
            f"{self.__class__.__name__} expected output value to contain "
            f"one of {list(self.values_map.keys())} (case-insensitive). "
            f"Received {cleaned_text}."
        )

    def parse(self, text: str) -> float:
        cleaned_text = text.split("\n")[-1].strip().lower()
        if self.parser_mode == "strict":
            return self._parse_strict(cleaned_text)
        if self.parser_mode == "loose":
            return self._parse_loose(cleaned_text)

        raise OutputParserException(f"Unknown parser_mode {self.parser_mode}.")
