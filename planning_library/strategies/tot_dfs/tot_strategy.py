from __future__ import annotations
from collections import deque
from typing import (
    AsyncIterator,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    Sequence,
)

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from ...action_executors import BaseActionExecutor, DefaultActionExecutor
from ..base_strategy import BaseCustomStrategy
from .components import BaseThoughtGenerator, BaseThoughtSorter, ThoughtEvaluator
from .components.thought_evaluators import (
    RunnableThoughtEvaluator,
    ThresholdThoughtEvaluatorContinueJudge,
)
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
    do_sorting: bool = (
        False  # True for DFS (Tree of Thoughts), False for DFSDT (ToolLLM)
    )
    root: Optional[ToTNode] = None
    terminals: List[ToTNode] = []

    @staticmethod
    def create(
        agent: Union[BaseSingleActionAgent, BaseMultiActionAgent],
        tools: Sequence[BaseTool],
        action_executor: Optional[BaseActionExecutor] = None,
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
            action_executor: The action executor for the current strategy. If None, the default will be used.
            evaluator_runnable: Runnable that powers ThoughtEvaluator. If None, the default model will be used.
            value_threshold: Threshold for evaluator; only thoughts evaluated higher than the threshold will be further explored.
            max_thoughts: Maximum number of new thoughts at each DFS step.
            max_iterations: Maximum number of iterations.
        """
        if evaluator_runnable is None:
            raise ValueError(
                "Default runnable for thought evaluator is not supported yet."
            )

        if action_executor is None:
            action_executor = DefaultActionExecutor(tools)

        strategy = TreeOfThoughtsDFSStrategy(
            agent=agent,
            thought_generator=AgentThoughtGenerator(),
            thought_evaluator=ThoughtEvaluator(
                backbone=RunnableThoughtEvaluator(evaluator_runnable),
                judge=ThresholdThoughtEvaluatorContinueJudge(value_threshold),
            ),
            max_thoughts=max_thoughts,
            max_iterations=max_iterations,
            action_executor=action_executor,
        )
        return strategy

    def _dfs_step(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[List[AgentAction] | AgentAction | AgentFinish]:
        """Performs a single step of DFS algorithm.

        Args:
            inputs: Agent inputs.
            trajectory: Current trajectory – path from the root node to the current node. Essentially, intermediate steps before the current DFS step.
            run_manager: Callback for the current run.

        Returns:
            Iterator over promising thoughts for the current step with three options possible:
              * List[AgentAction] - for multi-action thoughts
              * AgentAction - for single-action thoughts
              * AgentFinish - for finishing thoughts / thoughts without tool calls
        """

        # 1: generate k possible next steps
        thoughts = self.thought_generator.generate(
            agent=self.agent,
            inputs=inputs,
            trajectory=trajectory,
            max_num_thoughts=self.max_thoughts,
            run_manager=run_manager.get_child(tag="generate_thoughts")
            if run_manager
            else None,
        )

        # 2: (optional) sort them
        if self.do_sorting:
            assert (
                self.thought_sorter is not None
            ), "Sorting enabled, but thought sorter was not passed."
            thoughts = self.thought_sorter.sort_thoughts(
                thoughts=thoughts,
                inputs=inputs,
                trajectory=trajectory,
                run_manager=run_manager.get_child(tag="sort_thoughts")
                if run_manager
                else None,
            )

        for cur_thought in thoughts:
            # 3: evaluate each thought
            cur_thought_should_continue = self.thought_evaluator.evaluate(
                inputs=inputs,
                trajectory=trajectory,
                next_thought=cur_thought,
                run_manager=run_manager.get_child(tag="evaluate_thought")
                if run_manager
                else None,
            )

            # 4: proceed only with thoughts with value above a certain threshold
            if cur_thought_should_continue:
                yield cur_thought

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

            # TODO: traverses from the tree root to the cur_node on each call. how to optimize?
            trajectory = cur_node.trajectory

            for new_thought in self._dfs_step(
                inputs=inputs,
                trajectory=trajectory,
                run_manager=run_manager,
            ):
                # actually do action(s)
                if isinstance(new_thought, AgentFinish):
                    observation = None
                else:
                    observation = self.action_executor.execute(
                        actions=new_thought,
                        run_manager=run_manager.get_child() if run_manager else None,
                        reset_before_action=True,
                        trajectory=trajectory,
                    )

                new_node = ToTNode(
                    parent=cur_node, thought=new_thought, observation=observation
                )

                cur_node.children.append(new_node)
                if isinstance(new_thought, AgentFinish):
                    self.terminals.append(new_node)
                else:
                    frontier.appendleft(new_node)

            cur_step += 1

        for node in self.terminals:
            assert isinstance(
                node.thought, AgentFinish
            ), "Terminal nodes are expected to contain AgentFinish."
            yield node.thought, node.trajectory

    async def _adfs_step(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> AsyncIterator[List[AgentAction] | AgentAction | AgentFinish]:
        """Performs a single step of DFS algorithm asynchronously.

        Args:
            inputs: Agent inputs.
            trajectory: Current trajectory – path from the root node to the current node. Essentially, intermediate steps before the current DFS step.
            name_to_tool_map: Mapping from tool names to actual tools, used for calling tools based on agent's output.
            color_mapping: Mapping from tool names to colors, used for logging purposes when calling tools.
            run_manager: Callback for the current run.

        Returns:
            Iterator over promising thoughts for the current step with three options possible:
              * List[AgentAction] - for multi-action thoughts
              * AgentAction - for single-action thoughts
              * AgentFinish - for finishing thoughts / thoughts without tool calls
        """

        # 1: generate k possible next steps
        thoughts = await self.thought_generator.agenerate(
            agent=self.agent,
            inputs=inputs,
            trajectory=trajectory,
            max_num_thoughts=self.max_thoughts,
            run_manager=run_manager.get_child(tag="generate_thoughts")
            if run_manager
            else None,
        )

        # 2: (optional) sort them
        if self.do_sorting:
            assert (
                self.thought_sorter is not None
            ), "Sorting enabled, but thought sorter was not passed."
            thoughts = await self.thought_sorter.asort_thoughts(
                thoughts=thoughts,
                inputs=inputs,
                trajectory=trajectory,
                run_manager=run_manager.get_child(tag="sort_thoughts")
                if run_manager
                else None,
            )

        for cur_thought in thoughts:
            # 3: evaluate each thought
            cur_thought_should_continue = await self.thought_evaluator.aevaluate(
                inputs=inputs,
                trajectory=trajectory,
                next_thought=cur_thought,
                run_manager=run_manager.get_child(tag="evaluate_thought")
                if run_manager
                else None,
            )

            # 4: proceed only with thoughts with value above a certain threshold
            if cur_thought_should_continue:
                yield cur_thought

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

            # TODO: traverses from the tree root to the cur_node on each call. how to optimize?
            trajectory = cur_node.trajectory

            async for new_thought in self._adfs_step(
                inputs=inputs,
                trajectory=trajectory,
                run_manager=run_manager,
            ):
                # actually do action(s)
                if isinstance(new_thought, AgentFinish):
                    observation = None
                else:
                    observation = await self.action_executor.aexecute(
                        actions=new_thought,
                        run_manager=run_manager.get_child() if run_manager else None,
                        reset_before_action=True,
                        trajectory=trajectory,
                    )

                new_node = ToTNode(
                    parent=cur_node, thought=new_thought, observation=observation
                )
                cur_node.children.append(new_node)
                if isinstance(new_thought, AgentFinish):
                    self.terminals.append(new_node)
                else:
                    frontier.appendleft(new_node)

            cur_step += 1

        for node in self.terminals:
            assert isinstance(
                node.thought, AgentFinish
            ), "Terminal nodes are expected to contain AgentFinish."
            yield node.thought, node.trajectory
