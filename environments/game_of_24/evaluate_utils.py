from typing import Dict, Literal

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

_few_shot_evaluate_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "Input: {input}\nThought: {thought}\nJudge:"),
            ("ai", "{answer}"),
        ]
    ),
    examples=[
        {"input": "2 8 14", "thought": "2 + 8 = 10 (left: 10 14)", "answer": "10 + 14 = 24\nsure"},
        {
            "input": "6 8 14",
            "thought": "2 + 8 = 10 (left: 10 14)",
            "answer": "10 + 14 = 24\nbut there was no 2 in input\nimpossible",
        },
        {
            "input": "2 6 11",
            "thought": "2 * 6 = 12 (left: 11 12)",
            "answer": "11 + 12 = 23\n12 - 11 = 1\n11 * 12 = 132\n11 / 12 = 0.91\nimpossible",
        },
        {
            "input": "2 2 4 10",
            "thought": "2 + 2 = 4 (left: 4 4 10)",
            "answer": "4 + 4 + 10 = 8 + 10 = 18\n4 * 10 - 4 = 40 - 4 = 36\n(10 - 4) * 4 = 6 * 4 = 24\nsure",
        },
        {"input": "4 1 9 10", "thought": "10 + 1 = 11 (left: 4 9 11)", "answer": "9 + 11 + 4 = 20 + 4 = 24\nsure"},
        {
            "input": "5 2 7 4",
            "thought": "2 * 4 = 8 (left: 5 7 8)",
            "answer": "5 + 7 + 8 = 12 + 8 = 20\n(8 - 5) * 7 = 3 * 7 = 21\nI cannot obtain 24 now, but numbers are within a reasonable range\nlikely",
        },
        {
            "input": "5 6 2 3",
            "thought": "2 * 3 = 6 (left: 5 6 6)",
            "answer": "5 + 6 + 6 = 17\n(6 - 5) * 6 = 1 * 6 = 6\nI cannot obtain 24 now, but numbers are within a reasonable range\nlikely",
        },
        {
            "input": "11 2 10 5",
            "thought": "2 * 5 = 10 (left: 10 10 11)",
            "answer": "10 + 10 + 11 = 31\n(11 - 10) * 10 = 10\n10 10 10 are all too big\nimpossible",
        },
        {
            "input": "5 1 2 3",
            "thought": "5 - 2 = 3 (left: 1 3 3)",
            "answer": "1 * 3 * 3 = 9\n(1 + 3) * 3 = 12\n1 3 3 are all too small\nimpossible",
        },
        {"input": "2 8", "thought": "2 + 8 = 10 (left: 10)", "answer": "10 != 24\nimpossible"},
        {"input": "12", "thought": "12 + 12 = 24 (left: 24)", "answer": "there was only one number 12\nimpossible"},
        {"input": "36 5", "thought": "36 / 5 = 7.2 (left: 7.2)", "answer": "only integers are allowed\nimpossible"},
    ],
)

game_of_24_evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that judges whether given numbers can reach 24."),
        (
            "human",
            "Evaluate if given numbers can reach 24 (sure/likely/impossible)",
        ),
        _few_shot_evaluate_prompt,
        ("human", "Input: {inputs}\nThought: {thought}\nJudge:"),
    ]
)


_few_shot_last_step_evaluate_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "Input: {inputs}\nAnswer: {thought}\nJudge:"),
            ("ai", "{answer}"),
        ]
    ),
    examples=[
        {"inputs": "4 4 6 8", "thought": "(4 + 8) * (6 - 4) = 24", "answer": "sure"},
        {"inputs": "2 9 10 12", "thought": "2 * 12 * (10 - 9) = 24", "answer": "sure"},
        {"inputs": "4 9 10 13", "thought": "(13 - 9) * (10 - 4) = 24", "answer": "sure"},
        {"inputs": "4 4 6 8", "thought": "(4 + 8) * (6 - 4) + 1 = 25", "answer": "impossible"},
        {"inputs": "2 9 10 12", "thought": "2 * 12 * (10 - 9) = 24", "answer": "sure"},
        {"inputs": "2 9 10 12", "thought": "2 * (12 - 10) = 24", "answer": "impossible"},
        {"inputs": "4 9 10 13", "thought": "(13 - 4) * (10 - 9) = 24", "answer": "impossible"},
    ],
)

game_of_24_last_step_evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that judges whether answers to Game of 24 are correct."),
        (
            "human",
            "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.",
        ),
        _few_shot_last_step_evaluate_prompt,
        ("human", "Input: {inputs}\nAnswer: {thought}\nJudge:"),
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
