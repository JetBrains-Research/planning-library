from __future__ import annotations

from typing import List
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, AIMessage


def format_thought(
    thought: List[AgentAction] | AgentAction | AgentFinish,
) -> List[BaseMessage]:
    if isinstance(thought, list):
        messages = []
        for action in thought:
            messages.extend(format_thought(action))
        return messages
    elif isinstance(thought, AgentAction):
        return [
            AIMessage(
                content=f"Call tool `{thought.tool}` with arguments `{thought.tool_input}`"
            )
        ]
    elif isinstance(thought, AgentFinish):
        return [
            AIMessage(
                content=f"Finish execution with return values `{thought.return_values}`"
            )
        ]

    raise ValueError(f"Unexpected type for `thought`: {type(thought)}")


def format_thoughts(
    thoughts: List[List[AgentAction] | AgentAction | AgentFinish],
) -> List[BaseMessage]:
    messages = []
    for thought in thoughts:
        messages.extend(format_thought(thought))
    return messages
