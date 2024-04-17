from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from textwrap import dedent
from typing import Optional


from planning_library.components import AgentComponent
from typing import Tuple, Dict, Any, List, Sequence
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class ReflexionActorInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[Tuple[AgentAction, str]]
    self_reflections: Sequence[BaseMessage]


class ReflexionActor(AgentComponent[ReflexionActorInput]):
    """Actor component for Reflexion strategy.

    This component is powered by the agent and expects ReflexionActorInput to be passed as input.

    In addition to default agent initialization, this component also provides a default prompt template, which can
    be initialized with only a single user message.
    """

    name = "Actor"

    required_prompt_input_vars = set(ReflexionActorInput.__annotations__) - {
        "inputs",
        "intermediate_steps",
    } | {"agent_scratchpad"}

    @classmethod
    def _create_default_prompt(
        cls, system_message: Optional[str], user_message: str
    ) -> ChatPromptTemplate:
        if system_message is None:
            system_message = "You are an advanced reasoning agent that can improve based on self-reflection."

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    user_message,
                ),
                (
                    "human",
                    dedent("""
                        This might be not your first attempt. 
                        In this case, you will find self-reflective thoughts below. 
                        Make sure to pay extra attention to them, as they aim to identify and mitigate the exact shortcomings that led to failure in previous trials. 
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
