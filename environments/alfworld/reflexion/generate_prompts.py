from textwrap import dedent

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

generate_openai_tools_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are ALFRED, an intelligent agent navigating in a household."),
        (
            "human",
            dedent("""{inputs}"""),
        ),
        (
            "human",
            dedent("""This might be not your first attempt to fulfill the task. 
            In this case, you will find self-reflective thoughts below. Make sure to pay extra attention to them,
             as they aim to identify and mitigate the exact shortcomings that led to failure in previous trials. 
            """),
        ),
        MessagesPlaceholder("self_reflections"),
        (
            "human",
            "Good luck!",
        ),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
