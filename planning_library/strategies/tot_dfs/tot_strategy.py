import asyncio
from collections import deque
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from ...utils import aperform_agent_action, perform_agent_action
from ..base_strategy import BaseCustomStrategy
from .components import BaseThoughtGenerator, BaseThoughtSorter, ThoughtEvaluator
from .components.thought_evaluators import RunnableThoughtEvaluator, ThresholdThoughtEvaluatorContinueJudge
from .components.thought_generators import AgentThoughtGenerator
from .utils import ToTNode


class TreeOfThoughtsDFSStrategy(BaseCustomStrategy):
    thought_generator: BaseThoughtGenerator
    thought_evaluator: ThoughtEvaluator
    max_thoughts: int
    thought_sorter: Optional[BaseThoughtSorter] = None
    do_sorting: bool = False  # True for DFS (Tree-of-Thoughts), False for DFSDT (ToolLLM)
    root: Optional[ToTNode] = None
    terminals: List[ToTNode] = []

    @staticmethod
    def create(
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        tools: Sequence[BaseTool],
        evaluator_runnable: Optional[Runnable] = None,
        value_threshold: float = 0.5,
        max_thoughts: int = 3,
        max_iterations: int = 15,
        **kwargs,
    ) -> "TreeOfThoughtsDFSStrategy":
        if evaluator_runnable is None:
            raise ValueError("Default runnable for thought evaluator is not supported yet.")

        strategy = TreeOfThoughtsDFSStrategy(
            agent=agent,
            tools=tools,
            thought_generator=AgentThoughtGenerator(),
            thought_evaluator=ThoughtEvaluator(
                backbone=RunnableThoughtEvaluator(evaluator_runnable),
                judge=ThresholdThoughtEvaluatorContinueJudge(value_threshold),
            ),
            max_thoughts=max_thoughts,
            max_iterations=max_iterations,
        )
        return strategy

    def _perform_thought_actions(
        self,
        thought: Union[List[AgentAction], AgentAction, AgentFinish],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Optional[Union[List[AgentStep], AgentStep]]:
        if isinstance(thought, AgentAction):
            tool_result = perform_agent_action(
                agent_action=thought,
                name_to_tool_map=name_to_tool_map,
                color_mapping=color_mapping,
                verbose=self.verbose,
                tool_run_kwargs=self.agent.tool_run_logging_kwargs(),
                run_manager=run_manager,
            )
            return tool_result
        elif isinstance(thought, list):
            observations = []
            for action in thought:
                tool_result = perform_agent_action(
                    agent_action=action,
                    name_to_tool_map=name_to_tool_map,
                    color_mapping=color_mapping,
                    verbose=self.verbose,
                    tool_run_kwargs=self.agent.tool_run_logging_kwargs(),
                    run_manager=run_manager,
                )
                observations.append(tool_result)
            return observations
        return None

    async def _aperform_thought_actions(
        self,
        thought: Union[List[AgentAction], AgentAction, AgentFinish],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Optional[Union[List[AgentStep], AgentStep]]:
        if isinstance(thought, AgentAction):
            tool_result = await aperform_agent_action(
                agent_action=thought,
                name_to_tool_map=name_to_tool_map,
                color_mapping=color_mapping,
                verbose=self.verbose,
                tool_run_kwargs=self.agent.tool_run_logging_kwargs(),
                run_manager=run_manager,
            )
            return tool_result
        elif isinstance(thought, list):
            # TODO: no idea why mypy complains
            with asyncio.TaskGroup() as tg:  # type: ignore[attr-defined]
                tool_results = [
                    tg.create_task(
                        aperform_agent_action(
                            agent_action=action,
                            name_to_tool_map=name_to_tool_map,
                            color_mapping=color_mapping,
                            verbose=self.verbose,
                            tool_run_kwargs=self.agent.tool_run_logging_kwargs(),
                            run_manager=run_manager,
                        )
                    )
                    for action in thought
                ]
            return [task.result() for task in tool_results]
        return None

    def _dfs_step(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[
        Tuple[Union[List[AgentAction], AgentAction, AgentFinish], Optional[Union[List[AgentStep], AgentStep]]]
    ]:
        """Performs a single step of DFS algorithm."""

        # 1: generate k possible next steps
        thoughts = self.thought_generator.generate(
            agent=self.agent,
            inputs=inputs,
            trajectory=trajectory,
            max_num_thoughts=self.max_thoughts,
            run_manager=run_manager.get_child(tag="generate_thoughts") if run_manager else None,
        )

        # 2: (optional) sort them
        if self.do_sorting:
            assert self.thought_sorter is not None, "Sorting enabled, but thought sorter was not passed."
            thoughts = self.thought_sorter.sort_thoughts(
                thoughts=thoughts,
                inputs=inputs,
                trajectory=trajectory,
                run_manager=run_manager.get_child(tag="sort_thoughts") if run_manager else None,
            )

        for cur_thought in thoughts:
            # 3: do actions
            if isinstance(cur_thought, AgentFinish):
                observation = None
            else:
                observation = self._perform_thought_actions(
                    thought=cur_thought,
                    name_to_tool_map=name_to_tool_map,
                    color_mapping=color_mapping,
                    run_manager=run_manager,
                )
            # 4: evaluate each thought
            cur_thought_should_continue = self.thought_evaluator.evaluate(
                inputs=inputs,
                trajectory=trajectory,
                next_thought=cur_thought,
                observation=observation,
                run_manager=run_manager.get_child(tag="evaluate_thought") if run_manager else None,
            )

            # 5: proceed only with thoughts with value above a certain threshold
            if cur_thought_should_continue:
                yield cur_thought, observation

    def _run_strategy(
        self,
        inputs: Dict[str, str],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]:
        if not self.root:
            self.root = ToTNode()

        frontier: Deque[ToTNode] = deque([self.root])

        cur_step = 0
        while frontier and cur_step < self.max_iterations:
            cur_node = frontier.pop()

            for new_thought, observation in self._dfs_step(
                inputs=inputs,
                trajectory=cur_node.trajectory,
                run_manager=run_manager,
                name_to_tool_map=name_to_tool_map,
                color_mapping=color_mapping,
            ):
                new_node = ToTNode(parent=cur_node, thought=new_thought, observation=observation)
                cur_node.children.append(new_node)
                if isinstance(new_thought, AgentFinish):
                    self.terminals.append(new_node)
                else:
                    frontier.appendleft(new_node)

            cur_step += 1

        for node in self.terminals:
            assert isinstance(node.thought, AgentFinish), "Terminal nodes are expected to contain AgentFinish."
            yield node.thought, node.trajectory

    async def _arun_strategy(
        self,
        inputs: Dict[str, str],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Iterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]:
        raise NotImplementedError()
