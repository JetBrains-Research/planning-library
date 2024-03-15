from typing import Any, Dict, Optional, Sequence, Tuple

from langchain.agents.tools import InvalidTool
from langchain_core.agents import AgentAction, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping


def get_tools_maps(
    tools: Sequence[BaseTool],
) -> Tuple[Dict[str, BaseTool], Dict[str, str]]:
    name_to_tool_map = {tool.name: tool for tool in tools}
    color_mapping = get_color_mapping(
        [tool.name for tool in tools], excluded_colors=["green"]
    )
    return name_to_tool_map, color_mapping


def perform_agent_action(
    agent_action: AgentAction,
    name_to_tool_map: Dict[str, BaseTool],
    color_mapping: Dict[str, str],
    verbose: bool = True,
    tool_run_kwargs: Optional[Dict[str, Any]] = None,
    run_manager: Optional[CallbackManagerForChainRun] = None,
) -> AgentStep:
    """
    Copied from langchain.agents.agent.AgentExecutor._perform_agent_action with slight modifications.
    """

    tool_run_kwargs = {} if tool_run_kwargs is None else tool_run_kwargs

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


async def aperform_agent_action(
    name_to_tool_map: Dict[str, BaseTool],
    color_mapping: Dict[str, str],
    agent_action: AgentAction,
    verbose: bool = True,
    tool_run_kwargs: Optional[Dict[str, Any]] = None,
    run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
) -> AgentStep:
    """
    Copied from langchain.agents.agent.AgentExecutor._aperform_agent_action with slight modifications.
    """
    tool_run_kwargs = {} if tool_run_kwargs is None else tool_run_kwargs

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
