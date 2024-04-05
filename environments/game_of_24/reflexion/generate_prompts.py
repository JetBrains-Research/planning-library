from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from textwrap import dedent


openai_tools_generate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an advanced reasoning agent that plays Game of 24. You can improve based on self-reflection.",
        ),
        (
            "human",
            dedent("""
            You are given four numbers, and your goal is to obtain 24 from given numbers via basic arithmetic operations. 
            This might be not the first attempt you took, so pay attention to self-reflections about your previous failures. 
            When you're ready to answer, make sure to include a mathematical expression showing how to obtain 24 from given numbers, 
            for instance: '(2 + 2) * (12 / 2) = 24'.
            
            Inputs:
            {inputs}
            Self-reflections:
            {self_reflections}"""),
        ),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
