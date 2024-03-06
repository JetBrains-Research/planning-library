from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.agent import RunnableAgent, RunnableMultiActionAgent
from langchain.chains.base import Chain
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain_core.pydantic_v1 import root_validator
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping
from langgraph.pregel import Pregel  # type: ignore[import-untyped]


class BaseCustomStrategy(Chain, ABC):
    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent]
    tools: Sequence[BaseTool]
    return_intermediate_steps: bool = False
    max_iterations: int = 15
    verbose: bool = True

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        return self.agent.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key."""
        if self.return_intermediate_steps:
            return self.agent.return_values + ["intermediate_steps"]
        else:
            return self.agent.return_values

    @root_validator()
    def validate_tools(cls, values: Dict) -> Dict:
        """Validate that tools are compatible with agent."""
        agent = values["agent"]
        tools = values["tools"]
        allowed_tools = agent.get_allowed_tools()
        if allowed_tools is not None:
            if set(allowed_tools) != set([tool.name for tool in tools]):
                raise ValueError(
                    f"Allowed tools ({allowed_tools}) different than "
                    f"provided tools ({[tool.name for tool in tools]})"
                )
        return values

    @root_validator()
    def validate_return_direct_tool(cls, values: Dict) -> Dict:
        """Validate that tools are compatible with agent."""
        agent = values["agent"]
        tools = values["tools"]
        if isinstance(agent, BaseMultiActionAgent):
            for tool in tools:
                if tool.return_direct:
                    raise ValueError("Tools that have `return_direct=True` are not allowed " "in multi-action agents")
        return values

    @root_validator(pre=True)
    def validate_runnable_agent(cls, values: Dict) -> Dict:
        """Convert runnable to agent if passed in."""
        agent = values["agent"]
        if isinstance(agent, Runnable):
            try:
                output_type = agent.OutputType
            except Exception as _:
                multi_action = False
            else:
                multi_action = output_type == Union[List[AgentAction], AgentFinish]

            if multi_action:
                values["agent"] = RunnableMultiActionAgent(runnable=agent)
            else:
                values["agent"] = RunnableAgent(runnable=agent)
        return values

    @abstractmethod
    def _run_strategy(
        self,
        inputs: Dict[str, str],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]:
        ...

    @abstractmethod
    async def _arun_strategy(
        self,
        inputs: Dict[str, str],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Iterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]:
        ...

    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    async def _areturn(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            await run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping([tool.name for tool in self.tools], excluded_colors=["green", "red"])

        outputs = [
            self._return(output, intermediate_steps, run_manager=run_manager)
            for output, intermediate_steps in self._run_strategy(
                name_to_tool_map=name_to_tool_map,
                color_mapping=color_mapping,
                inputs=inputs,
                run_manager=run_manager,
            )
        ]

        return {"outputs": outputs}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping([tool.name for tool in self.tools], excluded_colors=["green"])

        _outputs = await self._arun_strategy(
            name_to_tool_map=name_to_tool_map,
            color_mapping=color_mapping,
            inputs=inputs,
            run_manager=run_manager,
        )

        outputs = []
        for _output, _intermediate_steps in _outputs:
            output = await self._areturn(_output, _intermediate_steps, run_manager=run_manager)
            outputs.append(output)

        return {"outputs": outputs}


class BaseLangGraphStrategy(ABC):
    @staticmethod
    @abstractmethod
    def create(
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent], tools: Sequence[BaseTool], **kwargs
    ) -> Pregel:
        ...
