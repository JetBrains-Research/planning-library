from typing import Dict, Optional

from langchain.agents.tools import InvalidTool
from langchain_core.agents import AgentAction, AgentStep
from langchain_core.callbacks import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain_core.tools import BaseTool


def _perform_agent_action(
    agent_action: AgentAction,
    name_to_tool_map: Dict[str, BaseTool],
    color_mapping: Dict[str, str],
    verbose,
    tool_run_kwargs,
    run_manager: Optional[CallbackManagerForChainRun] = None,
) -> AgentStep:
    """
    Copied from langchain.agents.agent.AgentExecutor._perform_agent_action with slight modifications.
    """

    if run_manager:
        run_manager.on_agent_action(agent_action, color="green")
    # Otherwise we lookup the tool
    if agent_action.tool in name_to_tool_map:
        tool = name_to_tool_map[agent_action.tool]
        return_direct = tool.return_direct
        color = color_mapping[agent_action.tool]
        if return_direct:
            tool_run_kwargs["llm_prefix"] = ""

        # We then call the tool on the tool input to get an observation
        observation = tool.run(
            agent_action.tool_input,
            verbose=verbose,
            color=color,
            callbacks=run_manager.get_child() if run_manager else None,
            **tool_run_kwargs,
        )
    else:
        observation = InvalidTool().run(
            {
                "requested_tool_name": agent_action.tool,
                "available_tool_names": list(name_to_tool_map.keys()),
            },
            verbose=verbose,
            color=None,
            callbacks=run_manager.get_child() if run_manager else None,
            **tool_run_kwargs,
        )
    return AgentStep(action=agent_action, observation=observation)


async def _aperform_agent_action(
    name_to_tool_map: Dict[str, BaseTool],
    color_mapping: Dict[str, str],
    agent_action: AgentAction,
    tool_run_kwargs,
    verbose,
    run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
) -> AgentStep:
    """
    Copied from langchain.agents.agent.AgentExecutor._aperform_agent_action with slight modifications.
    """
    if run_manager:
        await run_manager.on_agent_action(agent_action, verbose=verbose, color="green")
    # Otherwise we lookup the tool
    if agent_action.tool in name_to_tool_map:
        tool = name_to_tool_map[agent_action.tool]
        return_direct = tool.return_direct
        color = color_mapping[agent_action.tool]
        if return_direct:
            tool_run_kwargs["llm_prefix"] = ""
        # We then call the tool on the tool input to get an observation
        observation = await tool.arun(
            agent_action.tool_input,
            verbose=verbose,
            color=color,
            callbacks=run_manager.get_child() if run_manager else None,
            **tool_run_kwargs,
        )
    else:
        observation = await InvalidTool().arun(
            {
                "requested_tool_name": agent_action.tool,
                "available_tool_names": list(name_to_tool_map.keys()),
            },
            verbose=verbose,
            color=None,
            callbacks=run_manager.get_child() if run_manager else None,
            **tool_run_kwargs,
        )
    return AgentStep(action=agent_action, observation=observation)
