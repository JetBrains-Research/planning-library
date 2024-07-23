from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from planning_library.strategies.adapt.components import ADaPTExecutor, ADaPTPlanner

from ..base_strategy import BaseCustomStrategy


class ADaPTStrategy(BaseCustomStrategy):
    """ADaPT strategy.

    Based on "ADaPT: As-Needed Decomposition and Planning with Language Models" by Prasad et al.
    """

    executor: ADaPTExecutor
    planner: ADaPTPlanner
    max_depth: int

    @property
    def input_keys(self) -> List[str]:
        # TODO: define properly
        return []

    @property
    def output_keys(self) -> List[str]:
        # TODO: define properly
        return []

    def _adapt_step(
        self,
        inputs: Dict[str, Any],
        depth: int,
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[bool, AgentFinish, List[Tuple[AgentAction, str]]]:
        """Performs an iteration of ADaPT strategy.

        Args:
            current_task: The input for the current iteration. It can either be the original input or a subtask of a plan generated on a previous step.
            intermediate_steps: A list of actions taken before the current iteration.
            run_manager: Callback for the current run.
        """
        # 1: if we're too deep in task decomposition, finish early
        if depth > self.max_depth:
            return (
                False,
                AgentFinish(return_values={}, log="Maximum decomposition depth reached."),
                intermediate_steps,
            )

        # 2: run task through executor
        executor_output = self.executor.invoke(
            inputs,  # type: ignore
            run_manager=run_manager.get_child(tag=f"executor:depth_{depth}") if run_manager else None,
        )

        is_completed, cur_agent_outcome, cur_intermediate_steps = (
            executor_output["is_completed"],
            executor_output["agent_outcome"],
            executor_output["intermediate_steps"],
        )

        # 3.1: if executor estimated successful completion of a task, wrap up
        if is_completed:
            intermediate_steps.extend(cur_intermediate_steps)
            return True, cur_agent_outcome, intermediate_steps
        else:
            # 3.2: otherwise:
            self.executor.reset(
                actions=[a[0] for a in intermediate_steps],
                run_manager=run_manager.get_child(tag="clean_env") if run_manager else None,
            )

            # call a planner to further decompose a current task
            plan = self.planner.invoke(
                dict(
                    inputs=inputs,  # type: ignore[reportArgumentType]
                    executor_agent_outcome=cur_agent_outcome,
                    executor_intermediate_steps=cur_intermediate_steps,
                ),
                run_manager=run_manager.get_child(tag=f"executor:depth_{depth}") if run_manager else None,
            )
            # when AND logic is given, execute tasks sequentially
            if plan["aggregation_mode"] == "and":
                for task_inputs in plan["subtasks"]:
                    cur_is_completed, cur_agent_outcome, cur_intermediate_steps = self._adapt_step(
                        inputs={"inputs": task_inputs},
                        depth=depth + 1,
                        run_manager=run_manager,
                        intermediate_steps=intermediate_steps,
                    )

                    if not cur_is_completed:
                        agent_outcome = AgentFinish(
                            return_values=cur_agent_outcome.return_values,
                            log=f"Couldn't solve the task. Last log: {cur_agent_outcome.log}",
                        )
                        return False, agent_outcome, intermediate_steps
                    else:
                        intermediate_steps.extend(cur_intermediate_steps)

                agent_outcome = AgentFinish(return_values={}, log="Task solved successfully!")
                return True, agent_outcome, intermediate_steps
            elif plan["aggregation_mode"] == "or":
                for task_inputs in plan["subtasks"]:
                    (
                        cur_is_completed,
                        cur_agent_outcome,
                        cur_intermediate_steps,
                    ) = self._adapt_step(
                        inputs={"inputs": task_inputs},  # TODO: hard-coded inputs key is ugly
                        depth=depth + 1,
                        run_manager=run_manager,
                        intermediate_steps=intermediate_steps,
                    )

                    if cur_is_completed:
                        agent_outcome = AgentFinish(return_values={}, log="Task solved successfully!")
                        return True, agent_outcome, intermediate_steps
                    else:
                        intermediate_steps.extend(cur_intermediate_steps)

                agent_outcome = AgentFinish(
                    return_values=cur_agent_outcome.return_values,
                    log=f"Couldn't solve the task. Last log: {cur_agent_outcome.log}",
                )
                return False, agent_outcome, intermediate_steps

            raise NotImplementedError("Currently, only `and` and `or` aggregation logic is supported.")

    def _run_strategy(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]:
        _, agent_outcome, intermediate_steps = self._adapt_step(
            inputs=inputs,
            depth=0,
            run_manager=run_manager,
            intermediate_steps=[],
        )
        yield agent_outcome, intermediate_steps

    async def _adapt_astep(
        self,
        inputs: Dict[str, Any],
        depth: int,
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Tuple[bool, AgentFinish, List[Tuple[AgentAction, str]]]:
        """Performs an iteration of ADaPT strategy asynchronously.

        Args:
            inputs: The input for the current iteration. It can either be the original input or a subtask of a plan generated on a previous step.
            depth: Current decomposition step.
            intermediate_steps: A list of actions taken before the current iteration.
            run_manager: Callback for the current run.
        """
        # 1: if we're too deep in task decomposition, finish early
        if depth > self.max_depth:
            return (
                False,
                AgentFinish(return_values={}, log="Maximum decomposition depth reached."),
                intermediate_steps,
            )

        # 2: run task through executor
        executor_output = await self.executor.ainvoke(
            dict(
                inputs=inputs,  # type: ignore
            ),
            run_manager=run_manager.get_child(tag=f"executor:depth_{depth}") if run_manager else None,
        )
        is_completed, cur_agent_outcome, cur_intermediate_steps = (
            executor_output["is_completed"],
            executor_output["agent_outcome"],
            executor_output["intermediate_steps"],
        )

        # 3.1: if executor estimated successful completion of a task, wrap up
        if is_completed:
            intermediate_steps.extend(cur_intermediate_steps)
            return True, cur_agent_outcome, intermediate_steps
        else:
            # 3.2: otherwise:
            await self.executor.areset(
                actions=[a[0] for a in intermediate_steps],
                run_manager=run_manager.get_child(tag="clean_env") if run_manager else None,
            )

            plan = await self.planner.ainvoke(
                dict(
                    inputs=inputs,  # type: ignore[reportArgumentType]
                    executor_agent_outcome=cur_agent_outcome,
                    executor_intermediate_steps=cur_intermediate_steps,
                ),
                run_manager=run_manager.get_child(tag=f"executor:depth_{depth}") if run_manager else None,
            )
            # when AND logic is given, execute tasks sequentially
            if plan["aggregation_mode"] == "and":
                for task_inputs in plan["subtasks"]:
                    (
                        cur_is_completed,
                        cur_agent_outcome,
                        cur_intermediate_steps,
                    ) = await self._adapt_astep(
                        inputs={"inputs": task_inputs},
                        depth=depth + 1,
                        run_manager=run_manager,
                        intermediate_steps=intermediate_steps,
                    )

                    if not cur_is_completed:
                        agent_outcome = AgentFinish(
                            return_values=cur_agent_outcome.return_values,
                            log=f"Couldn't solve the task. Last log: {cur_agent_outcome.log}",
                        )
                        return False, agent_outcome, intermediate_steps
                    else:
                        intermediate_steps.extend(cur_intermediate_steps)

                agent_outcome = AgentFinish(return_values={}, log="Task solved successfully!")
                return True, agent_outcome, intermediate_steps
            elif plan["aggregation_mode"] == "or":
                for task_inputs in plan["subtasks"]:
                    (
                        cur_is_completed,
                        cur_agent_outcome,
                        cur_intermediate_steps,
                    ) = await self._adapt_astep(
                        inputs={"inputs": task_inputs},
                        depth=depth + 1,
                        run_manager=run_manager,
                        intermediate_steps=intermediate_steps,
                    )

                    if cur_is_completed:
                        agent_outcome = AgentFinish(return_values={}, log="Task solved successfully!")
                        return True, agent_outcome, intermediate_steps
                    else:
                        intermediate_steps.extend(cur_intermediate_steps)

                agent_outcome = AgentFinish(
                    return_values=cur_agent_outcome.return_values,
                    log=f"Couldn't solve the task. Last log: {cur_agent_outcome.log}",
                )
                return False, agent_outcome, intermediate_steps

            raise NotImplementedError("Currently, only `and` and `or` aggregation logic is supported.")

    async def _arun_strategy(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AsyncIterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]:
        _, agent_outcome, intermediate_steps = await self._adapt_astep(
            inputs=inputs,
            depth=0,
            run_manager=run_manager,
            intermediate_steps=[],
        )
        yield agent_outcome, intermediate_steps
