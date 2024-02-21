from typing import Union

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import BaseOutputParser

game_of_24_generate_reflexion_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an advanced reasoning agent that can improve based on self reflection."),
        (
            "human",
            "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. If you managed to obtain 24, output a final answer.",
        ),
        FewShotChatMessagePromptTemplate(
            example_prompt=ChatPromptTemplate.from_messages(
                [
                    (
                        "human",
                        "Input: {inputs}\nPrevious steps: {intermediate_steps}\nYour previous self-reflexions: {self_reflections}\n",
                    ),
                    ("ai", "{thought}"),
                ]
            ),
            examples=[
                {
                    "inputs": "2 8 8 14",
                    "intermediate_steps": "none",
                    "self_reflections": "none",
                    "thought": "2 + 8 = 10 (left: 8 10 14)",
                },
                {
                    "inputs": "4 6 8 8",
                    "intermediate_steps": "4 * 6 = 24 (left: 24 8 8)\n24 + 8 = 36 (left: 36 8)",
                    "self_reflections": "I made an error in my response, as I computed 16 + 6 = 24 incorrectly (in fact, 16 + 6 = 22) and did not use number 4 that was present in input. Next time, I should be more cautious and double-check that I am using all input numbers before providing an answer.",
                    "thought": "36 - 8 = 24\nanswer: 4 * 6 + 8 - 8 = 24",
                },
            ],
        ),
        (
            "human",
            "Input: {inputs}\nPrevious steps: {intermediate_steps}\nYour previous self-reflexions: {self_reflections}\n",
        ),
    ]
)


class GameOf24GenerateReflexionOutputParser(BaseOutputParser[Union[AgentAction, AgentFinish]]):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        lines = [line.strip() for line in text.split("\n")]

        if lines[-1].lower().startswith("answer:"):
            return AgentFinish({"output": lines[-1].lower()[len("answer: ") :]}, "Successfully reached 24.")

        return AgentAction(tool="simple_tool", tool_input=text, log=text)
