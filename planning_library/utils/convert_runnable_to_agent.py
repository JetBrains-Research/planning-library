from typing import List, Union

from langchain.agents.agent import RunnableAgent, RunnableMultiActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import Runnable


def convert_runnable_to_agent(agent: Runnable) -> Union[RunnableAgent, RunnableMultiActionAgent]:
    """Convert runnable to agent if passed in.

    Copied from langchain.agents.agent.AgentExecutor.validate_runnable_agent.
    """
    try:
        output_type = agent.OutputType
    except Exception as _:
        multi_action = False
    else:
        multi_action = output_type == Union[List[AgentAction], AgentFinish]

    if multi_action:
        return RunnableMultiActionAgent(runnable=agent)
    return RunnableAgent(runnable=agent)
