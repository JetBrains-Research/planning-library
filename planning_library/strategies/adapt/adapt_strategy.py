from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple, AsyncIterator, Any

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)


from ..base_strategy import BaseCustomStrategy
from planning_library.action_executors import LangchainActionExecutor, MetaTools
from planning_library.strategies.adapt.components import ADaPTExecutor, ADaPTPlanner
from planning_library.strategies.adapt.components.executor import ADaPTExecutorConfig
from planning_library.strategies.adapt.components.planner import ADaPTPlannerConfig
from dataclasses import asdict


class ADaPTStrategy(BaseCustomStrategy):
    """ADaPT strategy.

    Based on "ADaPT: As-Needed Decomposition and Planning with Language Models" by Prasad et al.
    """

    executor: ADaPTExecutor
    planner: ADaPTPlanner
    max_depth: int

    @staticmethod
    def create(
        meta_tools: Optional[MetaTools] = None,
        return_intermediate_steps: bool = False,
        return_finish_log: bool = False,
        max_iterations: int = 15,
        verbose: bool = True,
        executor_config: Optional[ADaPTExecutorConfig] = None,
        planner_config: Optional[ADaPTPlannerConfig] = None,
        max_depth: int = 20,
        **kwargs,
    ) -> "ADaPTStrategy":
        """Creates an instance of ADaPT strategy.

        Args:
            action_executor: The action executor for the current strategy. If None, the default will be used.
            executor_config: Configuration for the Executor component of ADaPT strategy.
            planner_config: Configuration for the Planner component of ADaPT strategy.
            return_intermediate_steps: True to additionally return a list of intermediate steps, False to simply return final outputs.
            return_finish_log: True to additionally return the finish log of the agent, False to simply return output values.
            max_iterations: Maximum allowed number of agent iterations for Executor and Planner.
            max_depth: Maximum depth of the ADaPT strategy (how deep the decomposition can go).
            verbose: True to print extra information during execution.
        """
        # TODO: runnable component vs strategy component?
        assert (
            executor_config is not None
        ), "Default ADaPT executor is currently not supported."

        if executor_config.runnable is not None:
            executor = ADaPTExecutor(
                executor_config.runnable,
                action_executor=LangchainActionExecutor(
                    tools=executor_config.tools if executor_config.tools else [],
                    meta_tools=executor_config.meta_tools,
                ),
            )
        else:
            executor = ADaPTExecutor.create_simple_strategy(
                **asdict(executor_config),
                return_intermediate_steps=return_intermediate_steps,
                return_finish_log=return_finish_log,
                max_iterations=max_iterations,
                verbose=verbose,
            )

        assert (
            planner_config is not None
        ), "Default ADaPT planner is currently not supported."

        if planner_config.runnable is not None:
            planner = ADaPTPlanner(planner_config.runnable)
        elif planner_config.mode == "agent":
            planner = ADaPTPlanner.create_agent_planner(
                **asdict(planner_config),
                executor_parser=executor_config.parser,
                executor_parser_name=executor_config.parser_name,
                return_intermediate_steps=return_intermediate_steps,
                return_finish_log=return_finish_log,
                max_iterations=max_iterations,
                verbose=verbose,
            )
        elif planner_config.mode == "simple":
            planner = ADaPTPlanner.create_simple_planner(
                **asdict(planner_config),
                executor_parser=executor_config.parser,
                executor_parser_name=executor_config.parser_name,
            )
        else:
            raise ValueError(
                f"Unknown planner mode `{planner_config.mode}`. Currently supported are: `agent`, `simple`."
            )

        strategy = ADaPTStrategy(
            executor=executor,
            planner=planner,
            max_depth=max_depth,
            return_intermediate_steps=return_intermediate_steps,
            return_finish_log=return_finish_log,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        return strategy

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
                AgentFinish(
                    return_values={}, log="Maximum decomposition depth reached."
                ),
                intermediate_steps,
            )

        # 2: run task through executor
        executor_output = self.executor.invoke(
            inputs,  # type: ignore[arg-type]
            run_manager=run_manager.get_child(tag=f"executor:depth_{depth}")
            if run_manager
            else None,
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
                run_manager=run_manager.get_child(tag="clean_env")
                if run_manager
                else None,
            )

            # call a planner to further decompose a current task
            plan = self.planner.invoke(
                dict(
                    inputs=inputs["inputs"],
                    executor_agent_outcome=cur_agent_outcome,
                    executor_intermediate_steps=cur_intermediate_steps,
                ),
                run_manager=run_manager.get_child(tag=f"executor:depth_{depth}")
                if run_manager
                else None,
            )
            # when AND logic is given, execute tasks sequentially
            if plan["aggregation_mode"] == "and":
                for task_inputs in plan["subtasks"]:
                    cur_is_completed, cur_agent_outcome, cur_intermediate_steps = (
                        self._adapt_step(
                            inputs={
                                "inputs": {"inputs": task_inputs}
                            },  # TODO: hard-coded inputs key is ugly
                            depth=depth + 1,
                            run_manager=run_manager,
                            intermediate_steps=intermediate_steps,
                        )
                    )

                    if not cur_is_completed:
                        agent_outcome = AgentFinish(
                            return_values=cur_agent_outcome.return_values,
                            log=f"Couldn't solve the task. Last log: {cur_agent_outcome.log}",
                        )
                        return False, agent_outcome, intermediate_steps
                    else:
                        intermediate_steps.extend(cur_intermediate_steps)

                agent_outcome = AgentFinish(
                    return_values={}, log="Task solved successfully!"
                )
                return True, agent_outcome, intermediate_steps
            elif plan["aggregation_mode"] == "or":
                for task_inputs in plan["subtasks"]:
                    (
                        cur_is_completed,
                        cur_agent_outcome,
                        cur_intermediate_steps,
                    ) = self._adapt_step(
                        inputs={
                            "inputs": {"inputs": task_inputs}
                        },  # TODO: hard-coded inputs key is ugly
                        depth=depth + 1,
                        run_manager=run_manager,
                        intermediate_steps=intermediate_steps,
                    )

                    if cur_is_completed:
                        agent_outcome = AgentFinish(
                            return_values={}, log="Task solved successfully!"
                        )
                        return True, agent_outcome, intermediate_steps
                    else:
                        intermediate_steps.extend(cur_intermediate_steps)

                agent_outcome = AgentFinish(
                    return_values=cur_agent_outcome.return_values,
                    log=f"Couldn't solve the task. Last log: {cur_agent_outcome.log}",
                )
                return False, agent_outcome, intermediate_steps

            raise NotImplementedError(
                "Currently, only `and` and `or` aggregation logic is supported."
            )

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
                AgentFinish(
                    return_values={}, log="Maximum decomposition depth reached."
                ),
                intermediate_steps,
            )

        # 2: run task through executor
        executor_output = await self.executor.ainvoke(
            dict(
                inputs=inputs,
            ),
            run_manager=run_manager.get_child(tag=f"executor:depth_{depth}")
            if run_manager
            else None,
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
                run_manager=run_manager.get_child(tag="clean_env")
                if run_manager
                else None,
            )

            plan = await self.planner.ainvoke(
                dict(
                    inputs=inputs,
                    executor_agent_outcome=cur_agent_outcome,
                    executor_intermediate_steps=cur_intermediate_steps,
                ),
                run_manager=run_manager.get_child(tag=f"executor:depth_{depth}")
                if run_manager
                else None,
            )
            # when AND logic is given, execute tasks sequentially
            if plan["aggregation_mode"] == "and":
                for task_inputs in plan["subtasks"]:
                    (
                        cur_is_completed,
                        cur_agent_outcome,
                        cur_intermediate_steps,
                    ) = await self._adapt_astep(
                        inputs={"inputs": {"inputs": task_inputs}},
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

                agent_outcome = AgentFinish(
                    return_values={}, log="Task solved successfully!"
                )
                return True, agent_outcome, intermediate_steps
            elif plan["aggregation_mode"] == "or":
                for task_inputs in plan["subtasks"]:
                    (
                        cur_is_completed,
                        cur_agent_outcome,
                        cur_intermediate_steps,
                    ) = await self._adapt_astep(
                        inputs={"inputs": {"inputs": task_inputs}},
                        depth=depth + 1,
                        run_manager=run_manager,
                        intermediate_steps=intermediate_steps,
                    )

                    if cur_is_completed:
                        agent_outcome = AgentFinish(
                            return_values={}, log="Task solved successfully!"
                        )
                        return True, agent_outcome, intermediate_steps
                    else:
                        intermediate_steps.extend(cur_intermediate_steps)

                agent_outcome = AgentFinish(
                    return_values=cur_agent_outcome.return_values,
                    log=f"Couldn't solve the task. Last log: {cur_agent_outcome.log}",
                )
                return False, agent_outcome, intermediate_steps

            raise NotImplementedError(
                "Currently, only `and` and `or` aggregation logic is supported."
            )

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
