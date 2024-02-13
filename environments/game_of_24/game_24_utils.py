from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages(
        [
            ("human", "Input: {inputs}"),
            ("ai", "{steps}\nAnswer: {answer}"),
        ]
    ),
    examples=[
        {
            "inputs": "4 4 6 8",
            "steps": "4 + 8 = 12 (left: 4 6 12)\n6 - 4 = 2 (left: 2 12)\n2 * 12 = 24 (left: 24)",
            "answer": "(6 - 4) * (4 + 8) = 24",
        },
        {
            "inputs": "2 9 10 12",
            "steps": "12 * 2 = 24 (left: 9 10 24)\n10 - 9 = 1 (left: 1 24)\n24 * 1 = 24 (left: 24)",
            "answer": "(12 * 2) * (10 - 9) = 24",
        },
        {
            "inputs": "4 9 10 13",
            "steps": "13 - 10 = 3 (left: 3 4 9)\n9 - 3 = 6 (left: 4 6)\n4 * 6 = 24 (left: 24)",
            "answer": "4 * (9 - (13 - 10)) = 24",
        },
        {
            "inputs": "1 4 8 8",
            "steps": "8 / 4 = 2 (left: 1 2 8)\n1 + 2 = 3 (left: 3 8)\n3 * 8 = 24 (left: 24)",
            "answer": "(1 + 8 / 4) * 8 = 24",
        },
        {
            "inputs": "5 5 5 9",
            "steps": "5 + 5 = 10 (left: 5 9 10)\n10 + 5 = 15 (left: 9 15)\n15 + 9 = 24 (left: 24)",
            "answer": "((5 + 5) + 5) + 9 = 24",
        },
    ],
)

game_of_24_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that plays Game of 24."),
        (
            "human",
            "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.",
        ),
        _few_shot_prompt,
        ("human", "Input: {inputs}\n"),
    ]
)
