from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import BaseOutputParser

self_reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an advanced reasoning agent that can self-reflect on their shortcomings when solving reasoning tasks.",
        ),
        (
            "human",
            "You will be given your previous trial in Game of 24, where you had to use basic arithmetic operations (+ - * /) with given numbers to obtain 24. You were unsuccessful. In a few sentences, diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.",
        ),
        ("human", "Input: {inputs}"),
        MessagesPlaceholder("intermediate_steps"),
        ("human", "Answer: {agent_outcome}\nSelf-reflection:"),
    ]
)


class GameOf24SelfReflexionOutputParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        return text
