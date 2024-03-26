from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from textwrap import dedent

self_reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an advanced reasoning agent that can self-reflect on their shortcomings when solving reasoning tasks.",
        ),
        (
            "human",
            dedent("""
            You will be given your previous trial in the household navigation.
            
            Input:
            {inputs}"""),
        ),
        ("human", "Your intermediate steps:"),
        MessagesPlaceholder("intermediate_steps"),
        (
            "human",
            dedent("""
        In this trial, you were unsuccessful. 
        In a few sentences, diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same shortcomings. 
        Use complete sentences."""),
        ),
    ]
)
