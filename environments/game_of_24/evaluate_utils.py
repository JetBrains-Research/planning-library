from typing import Dict

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

_few_shot_evaluate_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "Input: {thought}\nJudge:"),
            ("ai", "{answer}"),
        ]
    ),
    examples=[
        {"thought": "2 + 8 = 10 (left: 10 14)", "answer": "10 + 14 = 24\nsure"},
        {
            "thought": "2 * 6 = 12 (left: 11 12)",
            "answer": "11 + 12 = 23\n12 - 11 = 1\n11 * 12 = 132\n11 / 12 = 0.91\nimpossible",
        },
        {
            "thought": "2 + 2 = 4 (left: 4 4 10)",
            "answer": "4 + 4 + 10 = 8 + 10 = 18\n4 * 10 - 4 = 40 - 4 = 36\n(10 - 4) * 4 = 6 * 4 = 24\nsure",
        },
        {"thought": "10 + 1 = 11 (left: 4 9 11)", "answer": "9 + 11 + 4 = 20 + 4 = 24\nsure"},
        {
            "thought": "2 * 4 = 8 (left: 5 7 8)",
            "answer": "5 + 7 + 8 = 12 + 8 = 20\n(8 - 5) * 7 = 3 * 7 = 21\nI cannot obtain 24 now, but numbers are within a reasonable range\nlikely",
        },
        {
            "thought": "2 * 3 = 6 (left: 5 6 6)",
            "answer": "5 + 6 + 6 = 17\n(6 - 5) * 6 = 1 * 6 = 6\nI cannot obtain 24 now, but numbers are within a reasonable range\nlikely",
        },
        {
            "thought": "2 * 5 = 10 (left: 10 10 11)",
            "answer": "10 + 10 + 11 = 31\n(11 - 10) * 10 = 10\n10 10 10 are all too big\nimpossible",
        },
        {
            "thought": "5 - 2 = 3 (left: 1 3 3)",
            "answer": "1 * 3 * 3 = 9\n(1 + 3) * 3 = 12\n1 3 3 are all too small\nimpossible",
        },
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
        ("human", "Input: {thought}\nJudge:"),
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

    def parse(self, text: str) -> float:
        cleaned_text = text.split("\n")[-1].strip().lower()
        if cleaned_text not in self.values_map:
            raise OutputParserException(
                f"{self.__class__.__name__} expected output value to be "
                f"one of {self.values_map.keys()} (case-insensitive). "
                f"Received {cleaned_text}."
            )
        return self.values_map[cleaned_text]
