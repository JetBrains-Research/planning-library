from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from textwrap import dedent

plan_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an advanced reasoning agent that can create plans for reasoning tasks.",
        ),
        (
            "human",
            dedent("""
            The task at hand is a household navigation. An attempt to solve it directly has failed, and you need to create a clear and concise plan instead.
            
            You will be given the task description, all the intermediate steps that were taken during the last trial and the final outcome. 
            
            Task description:
            {inputs}"""),
        ),
        ("human", "Intermediate steps:"),
        MessagesPlaceholder("intermediate_steps"),
        ("human", "Final outcome: {agent_outcome}"),
        (
            "human",
            dedent("""
        Please, output a step-by-step plan for how to solve the task successfully. Use the following format:
        ```
        Step 1: <Description of first step>
        Step 2: <Description of second step>
        ...
        Step N: <Description of Nth step>
        ```
        """),
        ),
    ]
)
