from __future__ import annotations

from langchain_core.agents import AgentFinish, AgentAction, AgentStep
from langchain_core.callbacks import (
    CallbackManagerForChainRun,
    AsyncCallbackManagerForChainRun,
)

from planning_library.strategies import BaseCustomStrategy
from planning_library.action_executors import BaseActionExecutor, DefaultActionExecutor

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.tools import BaseTool
from typing import Union, Sequence, Optional, Dict, Iterator, Tuple, List, AsyncIterator


class SimpleStrategy(BaseCustomStrategy):
    """Simple strategy akin to langchain.agents.AgentExecutor:
    calls agent in a loop until either AgentFinish is produced or early stopping condition in reached."""

    _agent: BaseSingleActionAgent | BaseMultiActionAgent

    @property
    def agent(self) -> Union[BaseSingleActionAgent, BaseMultiActionAgent]:
        return self._agent

    @agent.setter
    def agent(self, agent: Union[BaseSingleActionAgent, BaseMultiActionAgent]):
        self._agent = agent

    @classmethod
    def create(
        cls,
        tools: Sequence[BaseTool],
        action_executor: Optional[BaseActionExecutor] = None,
        return_intermediate_steps: bool = False,
        return_finish_log: bool = False,
        max_iterations: int = 15,
        verbose: bool = True,
        agent: Optional[Union[BaseSingleActionAgent, BaseMultiActionAgent]] = None,
        **kwargs,
    ) -> "SimpleStrategy":
        if action_executor is None:
            action_executor = DefaultActionExecutor(tools=tools)

        if agent is None:
            raise ValueError("Default agent is currently not supported.")

        return SimpleStrategy(
            _agent=agent,
            action_executor=action_executor,
            return_intermediate_steps=return_intermediate_steps,
            return_finish_log=return_finish_log,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    def _run_strategy(
        self,
        inputs: Dict[str, str],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
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
                intermediate_steps.append(
                    (action_results.action, action_results.observation)
                )
            else:
                intermediate_steps.extend(
                    (_action_results.action, _action_results.observation)
                    for _action_results in action_results
                )

            cur_iteration += 1

        stopped_outcome = AgentFinish(
            {"output": "Agent stopped due to iteration limit."}, ""
        )

        yield stopped_outcome, intermediate_steps
        return

    async def _arun_strategy(
        self,
        inputs: Dict[str, str],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
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
                intermediate_steps.append(
                    (action_results.action, action_results.observation)
                )
            else:
                intermediate_steps.extend(
                    (_action_results.action, _action_results.observation)
                    for _action_results in action_results
                )

            cur_iteration += 1

        stopped_outcome = AgentFinish(
            {"output": "Agent stopped due to iteration limit."}, ""
        )

        yield stopped_outcome, intermediate_steps
        return
