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
    Sequence,
)

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from langchain_core.tools import BaseTool

from ...action_executors import BaseActionExecutor, DefaultActionExecutor
from ..base_strategy import BaseCustomStrategy

from .components import (
    ThoughtSorter,
    ThoughtGenerator,
    ThoughtGeneratorInput,
    ThoughtSorterConfig,
    ThoughtEvaluatorConfig,
    ThoughtGeneratorConfig,
    ThoughtEvaluator,
    ThoughtSorterInput,
    ThoughtEvaluatorInput,
)

from .utils import ToTNode


class TreeOfThoughtsDFSStrategy(BaseCustomStrategy):
    """Tree of Thoughts powered by Depth-First Search.

    Based on "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" by Yao et al.

    Also supports DFSDT from "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs" by Qin et al.
    """

    thought_generator: ThoughtGenerator
    thought_evaluator: ThoughtEvaluator
    thought_sorter: Optional[ThoughtSorter] = None
    do_sorting: bool = (
        False  # True for DFS (Tree of Thoughts), False for DFSDT (ToolLLM)
    )
    root: Optional[ToTNode] = None
    terminals: List[ToTNode] = []

    @property
    def agent(self):
        return self.thought_generator.agent.agent

    @classmethod
    def create(
        cls,
        tools: Sequence[BaseTool],
        action_executor: Optional[BaseActionExecutor] = None,
        return_intermediate_steps: bool = False,
        return_finish_log: bool = False,
        max_iterations: int = 15,
        verbose: bool = True,
        generator_config: Optional[ThoughtGeneratorConfig] = None,
        evaluator_config: Optional[ThoughtEvaluatorConfig] = None,
        sorter_config: Optional[ThoughtSorterConfig] = None,
        do_sorting: bool = False,
        **kwargs,
    ) -> "TreeOfThoughtsDFSStrategy":
        """Creates an instance of Tree of Thoughts + DFS strategy.

        Tree of Thoughts + DFS requires an evaluator component. The default setting is as follows:
            * evaluator is a runnable that accepts EvaluatorInput and returns a float in a 0-1 range;
            * evaluator judges whether a new thought should be explored or discarded based on the threshold;
              thought is explored further only when value is greater than the given threshold.

        Args:
            tools: The valid tools the agent can call.
            action_executor: The action executor for the current strategy. If None, the default will be used.
            max_thoughts: Maximum number of new thoughts at each DFS step.
            max_iterations: Maximum number of iterations.
        """
        if generator_config is None:
            raise ValueError(
                "Default thought generator config is currently not supported."
            )

        if evaluator_config is None:
            raise ValueError(
                "Default thought evaluator config is currently not supported."
            )

        if do_sorting and sorter_config is None:
            raise ValueError(
                "Default thought sorter config is currently not supported."
            )

        generator_config.tools = tools
        generator = ThoughtGenerator.create_from_config(generator_config)
        evaluator = ThoughtEvaluator.create_from_config(evaluator_config)
        sorter = ThoughtSorter.create_from_config(sorter_config) if do_sorting else None  # type: ignore[arg-type]

        if action_executor is None:
            action_executor = DefaultActionExecutor(tools)

        return cls(
            thought_generator=generator,
            thought_evaluator=evaluator,
            thought_sorter=sorter,
            do_sorting=do_sorting,
            action_executor=action_executor,
            return_intermediate_steps=return_intermediate_steps,
            return_finish_log=return_finish_log,
            max_iterations=max_iterations,
            verbose=verbose,
        )

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
        thoughts = self.thought_generator.invoke(
            ThoughtGeneratorInput(inputs=inputs, intermediate_steps=trajectory),
            run_manager=run_manager.get_child(tag="generate_thoughts")
            if run_manager
            else None,
        )

        # 2: (optional) sort them
        if self.do_sorting:
            assert (
                self.thought_sorter is not None
            ), "Sorting enabled, but thought sorter was not passed."
            thoughts = self.thought_sorter.invoke(
                ThoughtSorterInput(
                    thoughts=thoughts, inputs=inputs, intermediate_steps=trajectory
                ),
                run_manager=run_manager.get_child(tag="sort_thoughts")
                if run_manager
                else None,
            )

        for cur_thought in thoughts:
            # 3: evaluate each thought
            cur_thought_should_continue = self.thought_evaluator.invoke(
                ThoughtEvaluatorInput(
                    inputs=inputs,
                    intermediate_steps=trajectory,
                    next_thought=cur_thought,
                ),
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
        thoughts = await self.thought_generator.ainvoke(
            ThoughtGeneratorInput(inputs=inputs, intermediate_steps=trajectory),
            run_manager=run_manager.get_child(tag="generate_thoughts")
            if run_manager
            else None,
        )

        # 2: (optional) sort them
        if self.do_sorting:
            assert (
                self.thought_sorter is not None
            ), "Sorting enabled, but thought sorter was not passed."
            thoughts = await self.thought_sorter.ainvoke(
                ThoughtSorterInput(
                    thoughts=thoughts, inputs=inputs, intermediate_steps=trajectory
                ),
                run_manager=run_manager.get_child(tag="sort_thoughts")
                if run_manager
                else None,
            )

        for cur_thought in thoughts:
            # 3: evaluate each thought
            cur_thought_should_continue = await self.thought_evaluator.ainvoke(
                ThoughtEvaluatorInput(
                    inputs=inputs,
                    intermediate_steps=trajectory,
                    next_thought=cur_thought,
                ),
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
