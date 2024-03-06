from typing import Optional, Tuple

from langchain_core.agents import AgentAction
from langchain_core.callbacks import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langgraph.prebuilt import ToolExecutor  # type: ignore[import]


def execute_tools(
    agent_action: AgentAction,
    tool_executor: ToolExecutor,
    run_manager: Optional[CallbackManagerForChainRun] = None,
) -> Tuple[AgentAction, str]:
    if run_manager:
        run_manager.on_agent_action(agent_action, color="green")

    output = tool_executor.invoke(
        agent_action,
        callbacks=run_manager.get_child() if run_manager else None,
    )
    return agent_action, str(output)


async def aexecute_tools(
    agent_action: AgentAction,
    tool_executor: ToolExecutor,
    run_manager: Optional[CallbackManagerForChainRun] = None,
) -> Tuple[AgentAction, str]:
    if run_manager:
        run_manager.on_agent_action(agent_action, color="green")

    output = await tool_executor.ainvoke(
        agent_action,
        callbacks=run_manager.get_child() if run_manager else None,
    )
    return agent_action, str(output)
