from typing import Optional, Tuple

from langchain_core.agents import AgentAction
from langchain_core.callbacks import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langgraph.prebuilt import ToolExecutor  # type: ignore[import]


def execute_tools(
    agent_action: AgentAction,
    tool_executor: ToolExecutor,
) -> Tuple[AgentAction, str]:
    output = tool_executor.invoke(agent_action)
    return agent_action, str(output)


async def aexecute_tools(
    agent_action: AgentAction,
    tool_executor: ToolExecutor,
) -> Tuple[AgentAction, str]:
    output = await tool_executor.ainvoke(
        agent_action,
    )
    return agent_action, str(output)
