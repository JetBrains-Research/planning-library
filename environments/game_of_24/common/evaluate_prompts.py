from typing import Dict, Literal

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

_few_shot_evaluate_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "Input: {inputs}\nSteps taken: {trajectory}\nSuggestion: {next_thought} = {observation}\nJudge:"),
            ("ai", "{answer}"),
        ]
    ),
    examples=[
        {
            "inputs": "1 1 8 14",
            "trajectory": "; ".join(["1 + 1 = 2"]),
            "next_thought": "2 + 8",
            "observation": "10",
            "answer": "Available numbers after previous steps were: 2 8 14, after last suggestion: 10 14\n10 + 14 = 24\nsure",
        },
        {
            "inputs": "2 12 8 14",
            "trajectory": "; ".join(["12 / 2 = 6"]),
            "next_thought": "2 + 8",
            "observation": "10",
            "answer": "Available numbers after previous steps were: 6 8 14, but last suggestion uses 2, which is no longer in the input\nimpossible",
        },
        {
            "inputs": "24 6 12 11",
            "trajectory": "; ".join(["24 / 12 = 2"]),
            "next_thought": "2 * 6 = 12",
            "observation": "12",
            "answer": "Available numbers after previous steps were: 2 6 11, after last suggestion: 12 11\n11 + 12 = 23, 12 - 11 = 1, 11 * 12 = 132, 11 / 12 = 0.91\nimpossible",
        },
        {
            "inputs": "4 1 9 10",
            "trajectory": "none",
            "next_thought": "10 + 1",
            "observation": "11",
            "answer": "There were no previous steps, so available numbers were: 4 1 9 10. Numbers left after last suggestion: 4 9 11\n9 + 11 + 4 = 20 + 4 = 24\nsure",
        },
        {
            "inputs": "5 2 7 4",
            "trajectory": "none",
            "next_thought": "2 * 4",
            "observation": "8",
            "answer": "There were no previous steps, so available numbers were: 5 2 7 4. Numbers left after last suggestion: 5 7 8\n5 + 7 + 8 = 12 + 8 = 20\n(8 - 5) * 7 = 3 * 7 = 21\nI cannot obtain 24 now, but numbers are within a reasonable range\nlikely",
        },
        {
            "inputs": "11 2 10 5",
            "trajectory": "none",
            "next_thought": "2 * 5",
            "observation": "10",
            "answer": "There were no previous steps, so available numbers were: 11 2 10 5. Numbers left after last suggestion: 10 10 11\n10 + 10 + 11 = 31\n(11 - 10) * 10 = 10\n10 10 10 are all too big\nimpossible",
        },
        {
            "inputs": "5 1 2 3",
            "trajectory": "none",
            "next_thought": "5 - 2",
            "observation": "3",
            "answer": "There were no previous steps, so available numbers were: 5 1 2 3. Numbers left after last suggestion: 1 3 3\n1 * 3 * 3 = 9\n(1 + 3) * 3 = 12\n1 3 3 are all too small\nimpossible",
        },
        {
            "inputs": "1 1 4 2",
            "trajectory": "; ".join(["4 * 2 = 8", "1 + 1 = 2"]),
            "next_thought": "2 + 8",
            "observation": "10",
            "answer": "Available numbers after previous steps were: 1 1 4 2 -> 1 1 8 -> 2 8. Numbers left after last suggestion: 10\n10 != 24\nimpossible",
        },
        {
            "inputs": "2 2 2 5",
            "trajectory": "; ".join(["2 + 2 = 4", "5 - 2 = 3", "3 * 4 = 12"]),
            "next_thought": "12 + 12",
            "observation": "24",
            "answer": "Available numbers after previous steps were: 2 2 2 5 -> 4 2 5 -> 3 4 -> 12. Last suggestion uses two numbers 12, but there was only one left.\nimpossible",
        },
        {
            "inputs": "2 2 2 5",
            "trajectory": "none",
            "next_thought": "get_remaining_numbers",
            "observation": "2 2 2 5",
            "answer": "This thought doesn't alter the numbers list\nsure",
        },
    ],
)

game_of_24_evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that judges whether given numbers can reach 24."),
        (
            "human",
            "Given inputs and intermediate steps for Game of 24, evaluate if a new suggestion is correct and allows to reach 24. You are allowed to comment your decision, but make sure to always output one of the following words in the end: 'sure', 'likely', 'impossible'. Here are some examples:\n",
        ),
        _few_shot_evaluate_prompt,
        ("human", "Input: {inputs}\nSteps taken: {trajectory}\nSuggestion: {next_thought} = {observation}\nJudge:"),
    ]
)


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
