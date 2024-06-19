import asyncio
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

from langchain.chains.base import Chain
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from planning_library.action_executors.meta_tools import MetaTools


class BaseCustomStrategy(Chain, ABC):
    return_intermediate_steps: bool = False
    return_finish_log: bool = False
    max_iterations: int = 15
    verbose: bool = True

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
        meta_tools: Optional[MetaTools] = None,
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
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]: ...

    async def _arun_strategy(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AsyncIterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]:
        loop = asyncio.get_event_loop()

        sync_run_manager = run_manager.get_sync() if run_manager is not None else None
        result = await loop.run_in_executor(None, self._run_strategy, inputs, sync_run_manager)
        for item in result:
            yield item

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
            await run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
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

        outputs = [
            self._return(output, intermediate_steps, run_manager=run_manager)
            for output, intermediate_steps in self._run_strategy(
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

        _outputs = self._arun_strategy(
            inputs=inputs,
            run_manager=run_manager,
        )

        outputs = []
        async for _output, _intermediate_steps in _outputs:
            output = await self._areturn(_output, _intermediate_steps, run_manager=run_manager)
            outputs.append(output)

        return {key: [output[key] for output in outputs] for key in outputs[0]}


# TODO: what should the interface be?
class BaseLangGraphStrategy(ABC): ...
