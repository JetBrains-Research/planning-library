from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.agent import RunnableAgent, RunnableMultiActionAgent
from langchain.chains.base import Chain
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.pydantic_v1 import root_validator
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.pregel import Pregel  # type: ignore[import-untyped]

from planning_library.action_executors import BaseActionExecutor
from planning_library.utils.actions_utils import get_tools_maps


class BaseCustomStrategy(Chain, ABC):
    agent: Union[BaseSingleActionAgent, BaseMultiActionAgent]
    action_executor: BaseActionExecutor
    return_intermediate_steps: bool = False
    return_finish_log: bool = False
    max_iterations: int = 15
    verbose: bool = True

    @property
    def tools(self) -> Sequence[BaseTool]:
        return self.action_executor.tools

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

    @staticmethod
    @abstractmethod
    def create(
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        tools: Sequence[BaseTool],
        action_executor: Optional[BaseActionExecutor] = None,
        **kwargs,
    ) -> "BaseCustomStrategy": ...

    @abstractmethod
    def _run_strategy(
        self,
        inputs: Dict[str, str],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]: ...

    @abstractmethod
    def _arun_strategy(
        self,
        inputs: Dict[str, str],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AsyncIterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]: ...

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
        if self.return_finish_log:
            final_output["finish_log"] = output.log
        return final_output

    async def _areturn(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            await run_manager.on_agent_finish(
                output, color="green", verbose=self.verbose
            )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        if self.return_finish_log:
            final_output["finish_log"] = output.log
        return final_output

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        name_to_tool_map, color_mapping = get_tools_maps(self.tools)

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
        name_to_tool_map, color_mapping = get_tools_maps(self.tools)

        _outputs = self._arun_strategy(
            name_to_tool_map=name_to_tool_map,
            color_mapping=color_mapping,
            inputs=inputs,
            run_manager=run_manager,
        )

        outputs = []
        async for _output, _intermediate_steps in _outputs:
            output = await self._areturn(
                _output, _intermediate_steps, run_manager=run_manager
            )
            outputs.append(output)

        return {"outputs": outputs}


class BaseLangGraphStrategy(ABC):
    @staticmethod
    @abstractmethod
    def create(agent: Runnable, tools: Sequence[BaseTool], **kwargs) -> Pregel: ...
