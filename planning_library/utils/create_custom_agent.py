from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.agents import AgentAction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from typing import Sequence, Callable, Any, List, Tuple
from langchain.pydantic_v1 import BaseModel


class CustomAgentComponents(BaseModel):
    output_parser: BaseOutputParser
    format_intermediate_steps: Callable[
        [Sequence[Tuple[AgentAction, str]]], List[BaseMessage]
    ]
    convert_tool: Callable[[BaseTool], Any]


def create_custom_agent(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    components: CustomAgentComponents,
) -> Runnable:
    missing_vars = {"agent_scratchpad"}.difference(prompt.input_variables)
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    llm_with_tools = llm.bind(tools=[components.convert_tool(tool) for tool in tools])

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: components.format_intermediate_steps(
                x["intermediate_steps"]
            )
        )
        | prompt
        | llm_with_tools
        | components.output_parser
    )
    return agent
