from textwrap import dedent

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that judges whether the episodes of household navigation ended in failure or success.",
        ),
        (
            "human",
            dedent("""
            You will be given an input and a sequence of intermediate steps from one episode of household navigation. Your goal is to evaluate whether the episode ended in success or in failure.   

            Take your time and comment your decision, but make sure to always output either 0 or 1 in the end, where 0 would mean 'the episode ended in failure' and 1 would mean 'the episode ended in success'. 
            Use the following format: [[number]]."""),
        ),
        (
            "human",
            dedent(
                """Here is the input for the current episode:\n{inputs}\nHere are the intermediate steps:"""
            ),
        ),
        MessagesPlaceholder("intermediate_steps"),
        (
            "human",
            dedent(
                """Your verdict (ALWAYS output either 0 or 1 in double brackets like [[number]]):"""
            ),
        ),
    ]
)
