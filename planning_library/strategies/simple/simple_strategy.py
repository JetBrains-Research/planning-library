from __future__ import annotations

from typing import AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.tools import BaseTool

from planning_library.action_executors import (
    BaseActionExecutor,
    LangchainActionExecutor,
    MetaTools,
)

from ..base_strategy import BaseCustomStrategy


class SimpleStrategy(BaseCustomStrategy):
    """Simple strategy akin to langchain.agents.AgentExecutor:
    calls agent in a loop until either AgentFinish is produced or early stopping condition in reached."""

    action_executor: BaseActionExecutor
    agent: BaseSingleActionAgent | BaseMultiActionAgent

    @property
    def input_keys(self) -> List[str]:
        return self.agent.input_keys

    @property
    def output_keys(self) -> List[str]:
        if self.return_intermediate_steps:
            return self.agent.return_values + ["intermediate_steps"]
        else:
            return self.agent.return_values

    @classmethod
    def create(
        cls,
        meta_tools: Optional[MetaTools] = None,
        return_intermediate_steps: bool = False,
        return_finish_log: bool = False,
        max_iterations: int = 15,
        verbose: bool = True,
        action_executor: Optional[BaseActionExecutor] = None,
        tools: Optional[Sequence[BaseTool]] = None,
        agent: Optional[Union[BaseSingleActionAgent, BaseMultiActionAgent]] = None,
        **kwargs,
    ) -> "SimpleStrategy":
        tools = tools if tools is not None else []
        action_executor = (
            action_executor
            if action_executor is not None
            else LangchainActionExecutor(tools=tools, meta_tools=meta_tools)
        )

        if agent is None:
            raise ValueError("Default agent is currently not supported.")

        return SimpleStrategy(
            agent=agent,
            action_executor=action_executor,
            return_intermediate_steps=return_intermediate_steps,
            return_finish_log=return_finish_log,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    def _run_strategy(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]:
        intermediate_steps: List[Tuple[AgentAction, str]] = []

        cur_iteration = 0
        while self.max_iterations is None or cur_iteration < self.max_iterations:
            agent_outcome = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )

            if isinstance(agent_outcome, AgentFinish):
                yield agent_outcome, intermediate_steps
                return

            action_results = self.action_executor.execute(
                agent_outcome,
                run_manager=run_manager.get_child() if run_manager else None,
            )

            if isinstance(action_results, AgentStep):
                intermediate_steps.append((action_results.action, action_results.observation))
            else:
                intermediate_steps.extend(
                    (_action_results.action, _action_results.observation) for _action_results in action_results
                )

            cur_iteration += 1

        stopped_outcome = AgentFinish({"output": "Agent stopped due to iteration limit."}, "")

        yield stopped_outcome, intermediate_steps
        return

    async def _arun_strategy(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AsyncIterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]:
        intermediate_steps: List[Tuple[AgentAction, str]] = []

        cur_iteration = 0
        while self.max_iterations is None or cur_iteration < self.max_iterations:
            agent_outcome = await self.agent.aplan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )

            if isinstance(agent_outcome, AgentFinish):
                yield agent_outcome, intermediate_steps
                return

            action_results = await self.action_executor.aexecute(agent_outcome)

            if isinstance(action_results, AgentStep):
                intermediate_steps.append((action_results.action, action_results.observation))
            else:
                intermediate_steps.extend(
                    (_action_results.action, _action_results.observation) for _action_results in action_results
                )

            cur_iteration += 1

        stopped_outcome = AgentFinish({"output": "Agent stopped due to iteration limit."}, "")

        yield stopped_outcome, intermediate_steps
        return
