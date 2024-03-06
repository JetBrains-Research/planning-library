from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder

game_of_24_generate_openai_tools_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that plays Game of 24."),
        (
            "human",
            "Your end goal is to obtain 24 from given numbers via basic arithmetic operations with given numbers. Use {max_num_thoughts} of available tool(s) to suggest possible next step(s) as the next step from current state. Make sure to suggest exactly {max_num_thoughts} tool call(s), no more and no less.",
        ),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

game_of_24_few_shot_generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that plays Game of 24."),
        (
            "human",
            "Use basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. If you managed to obtain 24 in a proposed thought and there are no numbers left, output a final answer after the thought.",
        ),
        FewShotChatMessagePromptTemplate(
            example_prompt=ChatPromptTemplate.from_messages(
                [
                    (
                        "human",
                        "Input: {inputs}\nPrevious steps: {intermediate_steps}\n{max_num_thoughts} variant(s) for a possible next step:",
                    ),
                    ("ai", "{thought}"),
                ]
            ),
            examples=[
                {
                    "inputs": "2 8 8 14",
                    "intermediate_steps": "",
                    "max_num_thoughts": "8",
                    "thought": "2 + 8 = 10 (left: 8 10 14)\n8 / 2 = 4 (left: 4 8 14)\n14 + 2 = 16 (left: 8 8 16)\n2 * 8 = 16 (left: 8 14 16)\n8 - 2 = 6 (left: 6 8 14)\n14 - 8 = 6 (left: 2 6 8)\n14 /  2 = 7 (left: 7 8 8)\n14 - 2 = 12 (left: 8 8 12)",
                },
                {
                    "inputs": "12 2 8 6",
                    "intermediate_steps": "\n".join(["12 - 2 = 10 (left: 10 8 6)", "10 + 8 = 18 (left: 18 6)"]),
                    "max_num_thoughts": "2",
                    "thought": "18 + 6 = 24; answer: 12 - 2 + 8 + 6 = 24\n18 - 6 = 12",
                },
                {
                    "inputs": "12 2 8 6",
                    "intermediate_steps": "",
                    "max_num_thoughts": "3",
                    "thought": "12 * 2 = 24 (left: 24 8 6)\n8 / 2 = 4 (left: 12 4 6)\n6 + 8 = 14 (left: 12 2 14)",
                },
            ],
        ),
        (
            "human",
            "Input: {inputs}\nPrevious steps: {intermediate_steps}\n{max_num_thoughts} variant(s) for a possible next step:",
        ),
    ]
)
