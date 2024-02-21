from abc import ABC, abstractmethod
from typing import Sequence

from langchain.agents import BaseSingleActionAgent
from langchain_core.agents import AgentAction
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_executor import ToolExecutor  # type: ignore[import]

from planning_library.langgraph_version.utils import AgentState, aexecute_tools, execute_tools


class BaseActor(ABC):
    @abstractmethod
    def act(self, state: AgentState, **kwargs) -> AgentState:
        pass

    @abstractmethod
    async def aact(self, state: AgentState, **kwargs) -> AgentState:
        pass

    @abstractmethod
    def execute_tools(self, state: AgentState, **kwargs) -> AgentState:
        pass

    @abstractmethod
    async def aexecute_tools(self, state: AgentState, **kwargs) -> AgentState:
        pass


class AgentActor(BaseActor):
    def __init__(self, agent: BaseSingleActionAgent, tools: Sequence[BaseTool]):
        self.agent = agent
        self.tool_executor = ToolExecutor(tools)

    def act(self, state: AgentState, **kwargs) -> AgentState:
        # TODO: fix type warning
        agent_outcome = self.agent.invoke(  # type: ignore[attr-defined]
            {
                "intermediate_steps": state["intermediate_steps"],
                "inputs": state["inputs"],
                "self_reflections": state["self_reflections"],
            }
        )

        return {
            "inputs": state["inputs"],
            "agent_outcome": agent_outcome,
            "evaluator_score": state["evaluator_score"],
            "self_reflections": [],
            "intermediate_steps": [],
            "iteration": state["iteration"],
        }

    def execute_tools(self, state: AgentState, **kwargs) -> AgentState:
        assert isinstance(
            state["agent_outcome"], AgentAction
        ), "AgentAction has to be passed in order to execute tools."

        tools_outcome = execute_tools(state["agent_outcome"], self.tool_executor)

        return {
            "inputs": state["inputs"],
            "agent_outcome": state["agent_outcome"],
            "evaluator_score": state["evaluator_score"],
            "self_reflections": [],
            "intermediate_steps": [tools_outcome],
            "iteration": state["iteration"],
        }

    async def aact(self, state: AgentState, **kwargs) -> AgentState:
        # TODO: fix type warning
        agent_outcome = await self.agent.ainvoke(  # type: ignore[attr-defined]
            {
                "intermediate_steps": state["intermediate_steps"],
                "inputs": state["inputs"],
                "self_reflections": state["self_reflections"],
            }
        )

        return {
            "inputs": state["inputs"],
            "agent_outcome": agent_outcome,
            "evaluator_score": state["evaluator_score"],
            "self_reflections": [],
            "intermediate_steps": [],
            "iteration": state["iteration"],
        }

    async def aexecute_tools(self, state: AgentState, **kwargs) -> AgentState:
        assert isinstance(
            state["agent_outcome"], AgentAction
        ), "AgentAction has to be passed in order to execute tools."

        tools_outcome = await aexecute_tools(state["agent_outcome"], self.tool_executor)

        return {
            "inputs": state["inputs"],
            "agent_outcome": state["agent_outcome"],
            "evaluator_score": state["evaluator_score"],
            "self_reflections": [],
            "intermediate_steps": [tools_outcome],
            "iteration": state["iteration"],
        }
