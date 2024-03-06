from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder

_few_shot_evaluate_prompt = FewShotChatMessagePromptTemplate(
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

evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that judges whether answers to Game of 24 are correct."),
        (
            "human",
            "Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. 1) it uses each given number exactly once; 2) it doesn't use any other number; 3) given mathematical expression correctly reaches 24. Here are some examples:\n",
        ),
        _few_shot_evaluate_prompt,
        ("human", "Inputs: {inputs}"),
        MessagesPlaceholder("intermediate_steps"),
        ("human", "Answer: {agent_outcome}\nJudge:"),
    ]
)
