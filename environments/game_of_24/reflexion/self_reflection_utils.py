from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import BaseOutputParser

_few_shot_self_reflection_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "Input: {inputs}\nPrevious trial:\n{intermediate_steps}\nAnswer: {answer}\nReflection:"),
            ("ai", "{reflection}"),
        ]
    ),
    examples=[
        {
            "inputs": "4 6 8 10",
            "intermediate_steps": "2 + 8 = 10 (left: 10 10 4)\n10 + 10 = 20 (left: 20 4)\n20 + 4 = 24",
            "answer": "2 + 8 + 10 + 4 = 24",
            "reflection": "I made an error in my response, as I used number 2 that was not present in input. Next time, I should be more cautious and double-check that I am only using input numbers before providing an answer.",
        },
        {
            "inputs": "4 6 8 10",
            "intermediate_steps": "10 + 8 = 18 (left: 18 4 6)\n18 + 6 = 24 (left: 24 4)",
            "answer": "10 + 8 + 6 = 24",
            "reflection": "I made an error in my response, as I did not use number 4 that was present in input. Next time, I should be more cautious and double-check that I am using all input numbers before providing an answer.",
        },
        {
            "inputs": "4 6 8 10",
            "intermediate_steps": "10 + 8 = 14 (left: 14 4 6)\n14 + 6 = 20 (left: 20 4)\n20 + 4 = 24",
            "answer": "10 + 8 + 6 + 4 = 24",
            "reflection": "I made an error in my response, as I incorrectly computed that 10 + 8 = 14, while, in fact, 10 + 8 = 18. Next time, I should be more cautious and double-check the results of arithmetical operations before providing an answer.",
        },
    ],
)

game_of_24_self_reflection_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an advanced reasoning agent that can improve based on self reflection."),
        (
            "human",
            "You will be given a previous trial in Game of 24, where you had to use basic arithmetic operations (+ - * /) with given numbers to obtain 24. You were unsuccessful. In a few sentences, diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.\nHere are some examples:",
        ),
        _few_shot_self_reflection_prompt,
        ("human", "Input: {inputs}\nPrevious trial: {intermediate_steps}\nAnswer: {answer}\nReflection:"),
    ]
)


class GameOf24SelfReflexionOutputParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        return text
