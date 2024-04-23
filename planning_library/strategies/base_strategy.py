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
)

from langchain.chains.base import Chain
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.tools import BaseTool

from planning_library.action_executors import BaseActionExecutor
from planning_library.utils.actions_utils import get_tools_maps


class BaseCustomStrategy(Chain, ABC):
    action_executor: BaseActionExecutor
    return_intermediate_steps: bool = False
    return_finish_log: bool = False
    max_iterations: int = 15
    verbose: bool = True

    @property
    def tools(self) -> Sequence[BaseTool]:
        return self.action_executor.tools

    @property
    @abstractmethod
    def input_keys(self) -> List[str]:
        """Return the input keys."""
        ...

    @property
    @abstractmethod
    def output_keys(self) -> List[str]:
        """Return the singular output key."""
        ...

    @classmethod
    @abstractmethod
    def create(
        cls,
        action_executor: Optional[BaseActionExecutor] = None,
        return_intermediate_steps: bool = False,
        return_finish_log: bool = False,
        max_iterations: int = 15,
        verbose: bool = True,
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

        return {key: [output[key] for output in outputs] for key in outputs[0]}

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

        return {key: [output[key] for output in outputs] for key in outputs[0]}


# TODO: what should the interface be?
class BaseLangGraphStrategy(ABC): ...
