from typing import Dict, Iterator, List, Optional, Tuple, Union, Sequence, AsyncIterator

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from ...action_executors import BaseActionExecutor
from ..base_strategy import BaseCustomStrategy
from .components import BaseADaPTExecutor, BaseADaPTPlanner
from planning_library.strategies.adapt.utils import ADaPTTask
from planning_library.action_executors import DefaultActionExecutor
from planning_library.strategies.adapt.components.executors import StrategyADaPTExecutor
from planning_library.strategies.adapt.components.planners import RunnableADaPTPlanner
from planning_library.strategies.simple import SimpleStrategy


class ADaPTStrategy(BaseCustomStrategy):
    """ADaPT strategy.

    Based on "ADaPT: As-Needed Decomposition and Planning with Language Models" by Prasad et al.
    """

    executor: BaseADaPTExecutor
    planner: BaseADaPTPlanner
    max_depth: int

    @staticmethod
    def create(
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        tools: Sequence[BaseTool],
        action_executor: Optional[BaseActionExecutor] = None,
        planner_runnable: Optional[Runnable] = None,
        executor_strategy: Optional[BaseCustomStrategy] = None,
        max_depth: int = 20,
        **kwargs,
    ) -> "ADaPTStrategy":
        """Creates an instance of ADaPT strategy.

        Args:
            agent: The agent to run for proposing thoughts at each DFS step.
            tools: The valid tools the agent can call.
            action_executor: The action executor for the current strategy. If None, the default will be used.
            planner_runnable: Runnable that powers ADaPT planner. If None, the default model will be used.
            executor_strategy: Strategy that powers ADAPT executor. If None, the default model will be used.
            max_depth: Maximum depth of the ADaPT strategy (how deep the decomposition can go).
        """
        executor_kwargs = {
            kwarg[len("executor_") :]: kwargs[kwarg]
            for kwarg in kwargs
            if kwarg.startswith("executor_")
        }
        adapt_kwargs = {
            kwarg: kwargs[kwarg]
            for kwarg in kwargs
            if not kwarg.startswith("executor_")
        }

        if planner_runnable is None:
            raise ValueError("Default runnable for ADaPT planner is not supported yet.")

        if action_executor is None:
            action_executor = DefaultActionExecutor(tools)

        if executor_strategy is None:
            executor_strategy = SimpleStrategy.create(
                agent=agent,
                tools=tools,
                action_executor=action_executor,
                **executor_kwargs,
            )

        strategy = ADaPTStrategy(
            agent=agent,
            action_executor=action_executor,
            executor=StrategyADaPTExecutor(strategy=executor_strategy),
            planner=RunnableADaPTPlanner(runnable=planner_runnable),
            max_depth=max_depth,
            **adapt_kwargs,
        )
        return strategy

    def _adapt_step(
        self,
        current_task: ADaPTTask,
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
        if current_task["depth"] > self.max_depth:
            return (
                False,
                AgentFinish(
                    return_values={}, log="Maximum decomposition depth reached."
                ),
                intermediate_steps,
            )

        # 2: run task through executor
        is_completed, cur_agent_outcome, cur_intermediate_steps = self.executor.execute(
            inputs=current_task["inputs"],
            run_manager=run_manager.get_child(
                tag=f"executor:depth_{current_task['depth']}"
            )
            if run_manager
            else None,
        )

        # 3.1: if executor estimated successful completion of a task, wrap up
        if is_completed:
            intermediate_steps.extend(cur_intermediate_steps)
            return True, cur_agent_outcome, intermediate_steps
        else:
            # 3.2: otherwise:
            # clean up the environment
            self.action_executor.reset(
                actions=[step[0] for step in intermediate_steps],
                run_manager=run_manager.get_child() if run_manager else None,
            )

            # call a planner to further decompose a current task
            plan = self.planner.plan(
                inputs=current_task["inputs"],
                current_depth=current_task["depth"],
                agent_outcome=cur_agent_outcome,
                intermediate_steps=cur_intermediate_steps,
                run_manager=run_manager.get_child(
                    tag=f"executor:depth_{current_task['depth']}"
                )
                if run_manager
                else None,
            )
            # when AND logic is given, execute tasks sequentially
            if plan["logic"] == "and":
                for task in plan["subtasks"]:
                    cur_is_completed, cur_agent_outcome, cur_intermediate_steps = (
                        self._adapt_step(
                            current_task=task,
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

            raise NotImplementedError("Currently, only `and` logic is supported.")

    def _run_strategy(
        self,
        inputs: Dict[str, str],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]:
        _, agent_outcome, intermediate_steps = self._adapt_step(
            current_task={"inputs": inputs, "depth": 0},
            run_manager=run_manager,
            intermediate_steps=[],
        )
        yield agent_outcome, intermediate_steps

    async def _adapt_astep(
        self,
        current_task: ADaPTTask,
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Tuple[bool, AgentFinish, List[Tuple[AgentAction, str]]]:
        """Performs an iteration of ADaPT strategy asynchronously.

        Args:
            current_task: The input for the current iteration. It can either be the original input or a subtask of a plan generated on a previous step.
            intermediate_steps: A list of actions taken before the current iteration.
            run_manager: Callback for the current run.
        """
        # 1: if we're too deep in task decomposition, finish early
        if current_task["depth"] > self.max_depth:
            return (
                False,
                AgentFinish(
                    return_values={}, log="Maximum decomposition depth reached."
                ),
                intermediate_steps,
            )

        # 2: run task through executor
        (
            is_completed,
            cur_agent_outcome,
            cur_intermediate_steps,
        ) = await self.executor.aexecute(
            inputs=current_task["inputs"],
            run_manager=run_manager.get_child(
                tag=f"executor:depth_{current_task['depth']}"
            )
            if run_manager
            else None,
        )

        # 3.1: if executor estimated successful completion of a task, wrap up
        if is_completed:
            intermediate_steps.extend(cur_intermediate_steps)
            return True, cur_agent_outcome, intermediate_steps
        else:
            # 3.2: otherwise:
            # clean up the environment
            self.action_executor.reset(actions=[step[0] for step in intermediate_steps])

            plan = await self.planner.aplan(
                inputs=current_task["inputs"],
                current_depth=current_task["depth"],
                agent_outcome=cur_agent_outcome,
                intermediate_steps=cur_intermediate_steps,
                run_manager=run_manager.get_child(
                    tag=f"executor:depth_{current_task['depth']}"
                )
                if run_manager
                else None,
            )
            # when AND logic is given, execute tasks sequentially
            if plan["logic"] == "and":
                for task in plan["subtasks"]:
                    (
                        cur_is_completed,
                        cur_agent_outcome,
                        cur_intermediate_steps,
                    ) = await self._adapt_astep(
                        current_task=task,
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

            raise NotImplementedError("Currently, only `and` logic is supported.")

    async def _arun_strategy(
        self,
        inputs: Dict[str, str],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AsyncIterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]:
        _, agent_outcome, intermediate_steps = await self._adapt_astep(
            current_task={"inputs": inputs, "depth": 0},
            run_manager=run_manager,
            intermediate_steps=[],
        )
        yield agent_outcome, intermediate_steps
