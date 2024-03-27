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
            "Use observations from an environment to judge whether you have solved the task successfully. "
            "When you are sure that you have successfully completed the task, make sure to include the words 'task completed' in your output. "
            "Do not write 'task completed' if the task has not been completed.",
        ),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
