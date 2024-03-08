import asyncio
from collections import deque
from typing import AsyncIterator, Deque, Dict, Iterator, List, Optional, Sequence, Tuple, Union

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
    """Tree of Thoughts powered by Depth-First Search.

    Based on "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" by Yao et al.

    Also supports DFSDT from "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs" by Qin et al.
    """

    thought_generator: BaseThoughtGenerator
    thought_evaluator: ThoughtEvaluator
    max_thoughts: int
    thought_sorter: Optional[BaseThoughtSorter] = None
    do_sorting: bool = False  # True for DFS (Tree of Thoughts), False for DFSDT (ToolLLM)
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
        """Creates an instance of Tree of Thoughts + DFS strategy.

        Tree of Thoughts + DFS requires an evaluator component. The default setting is as follows:
            * evaluator is a runnable that accepts EvaluatorInput and returns a float in a 0-1 range;
            * evaluator judges whether a new thought should be explored or discarded based on the threshold;
              thought is explored further only when value is greater than the given threshold.

        Args:
            agent: The agent to run for proposing thoughts at each DFS step.
            tools: The valid tools the agent can call.
            evaluator_runnable: Runnable that powers ThoughtEvaluator. If None, the default model will be used.
            value_threshold: Threshold for evaluator; only thoughts evaluated higher than the threshold will be further explored.
            max_thoughts: Maximum number of new thoughts at each DFS step.
            max_iterations: Maximum number of iterations.
        """
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
        """Performs actions proposed as a thought.

        Args:
            thought: Actions proposed as a thought. Can be: multi-action, single action, finishing.
            name_to_tool_map: Mapping from tool names to actual tools, used for calling tools based on agent's output.
            color_mapping: Mapping from tool names to colors, used for logging purposes when calling tools.
            run_manager: Callback for the current run.

        Returns:
              * List[AgentStep] - for multi-action thoughts (List[AgentAction])
              * AgentStep - for single-action thoughts (AgentAction)
              * None - for finishing thoughts (AgentFinish)
        """
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
        """Performs actions proposed as a thought asynchronously.

        Args:
            thought: Actions proposed as a thought. Can be: multi-action, single action, finishing.
            name_to_tool_map: Mapping from tool names to actual tools, used for calling tools based on agent's output.
            color_mapping: Mapping from tool names to colors, used for logging purposes when calling tools.
            run_manager: Callback for the current run.

        Returns:
              * List[AgentStep] - for multi-action thoughts (List[AgentAction])
              * AgentStep - for single-action thoughts (AgentAction)
              * None - for finishing thoughts (AgentFinish)
        """
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
        """Performs a single step of DFS algorithm.

        Args:
            inputs: Agent inputs.
            trajectory: Current trajectory – path from the root node to the current node. Essentially, intermediate steps before the current DFS step.
            name_to_tool_map: Mapping from tool names to actual tools, used for calling tools based on agent's output.
            color_mapping: Mapping from tool names to colors, used for logging purposes when calling tools.
            run_manager: Callback for the current run.

        Returns:
            Iterator over tuples with three options possible:
              * tuple (List[AgentAction], List[AgentStep]) - for multi-action thoughts
              * tuple (AgentAction, AgentStep) - for single-action thoughts
              * tuple (AgentFinish, None) - for finishing thoughts
        """

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
        """Runs Tree of Thoughts + DFS strategy.

        Args:
            inputs: Agent inputs.
            name_to_tool_map: Mapping from tool names to actual tools, used for calling tools based on agent's output.
            color_mapping: Mapping from tool names to colors, used for logging purposes when calling tools.
            run_manager: Callback for the current run.

        Returns:
            Iterator over tuples (AgentFinish, List[Tuple[AgentAction, str]]):
            essentially, each tuple consists of the final result and of intermediate steps.
            The current implementation iterates over ALL terminal nodes in a tree.
        """
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

    async def _adfs_step(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AsyncIterator[
        Tuple[Union[List[AgentAction], AgentAction, AgentFinish], Optional[Union[List[AgentStep], AgentStep]]]
    ]:
        """Performs a single step of DFS algorithm asynchronously.

        Args:
            inputs: Agent inputs.
            trajectory: Current trajectory – path from the root node to the current node. Essentially, intermediate steps before the current DFS step.
            name_to_tool_map: Mapping from tool names to actual tools, used for calling tools based on agent's output.
            color_mapping: Mapping from tool names to colors, used for logging purposes when calling tools.
            run_manager: Callback for the current run.

        Returns:
            Iterator over tuples with three options possible:
              * tuple (List[AgentAction], List[AgentStep]) - for multi-action thoughts
              * tuple (AgentAction, AgentStep) - for single-action thoughts
              * tuple (AgentFinish, None) - for finishing thoughts
        """

        # 1: generate k possible next steps
        thoughts = await self.thought_generator.agenerate(
            agent=self.agent,
            inputs=inputs,
            trajectory=trajectory,
            max_num_thoughts=self.max_thoughts,
            run_manager=run_manager.get_child(tag="generate_thoughts") if run_manager else None,
        )

        # 2: (optional) sort them
        if self.do_sorting:
            assert self.thought_sorter is not None, "Sorting enabled, but thought sorter was not passed."
            thoughts = await self.thought_sorter.asort_thoughts(
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
                observation = await self._aperform_thought_actions(
                    thought=cur_thought,
                    name_to_tool_map=name_to_tool_map,
                    color_mapping=color_mapping,
                    run_manager=run_manager,
                )
            # 4: evaluate each thought
            cur_thought_should_continue = await self.thought_evaluator.aevaluate(
                inputs=inputs,
                trajectory=trajectory,
                next_thought=cur_thought,
                observation=observation,
                run_manager=run_manager.get_child(tag="evaluate_thought") if run_manager else None,
            )

            # 5: proceed only with thoughts with value above a certain threshold
            if cur_thought_should_continue:
                yield cur_thought, observation

    async def _arun_strategy(
        self,
        inputs: Dict[str, str],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AsyncIterator[Tuple[AgentFinish, List[Tuple[AgentAction, str]]]]:
        """Runs Tree of Thoughts + DFS strategy asynchronously.

        Args:
            inputs: Agent inputs.
            name_to_tool_map: Mapping from tool names to actual tools, used for calling tools based on agent's output.
            color_mapping: Mapping from tool names to colors, used for logging purposes when calling tools.
            run_manager: Callback for the current run.

        Returns:
            Iterator over tuples (AgentFinish, List[Tuple[AgentAction, str]]):
            essentially, each tuple consists of the final result and of intermediate steps.
            The current implementation iterates over ALL terminal nodes in a tree.
        """
        if not self.root:
            self.root = ToTNode()

        frontier: Deque[ToTNode] = deque([self.root])

        cur_step = 0
        while frontier and cur_step < self.max_iterations:
            cur_node = frontier.pop()

            async for new_thought, observation in self._adfs_step(
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
