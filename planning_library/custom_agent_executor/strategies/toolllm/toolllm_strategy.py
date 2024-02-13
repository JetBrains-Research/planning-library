from typing import Dict, List, Optional, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.tools import BaseTool

from ..base_strategy import BaseStrategy
from .thought_evaluators import BaseThoughtEvaluator
from .thought_generators import BaseThoughtGenerator
from .thought_sorters import BaseThoughtSorter


class TreeOfThoughtsDFSStrategy(BaseStrategy):
    thought_generator: BaseThoughtGenerator
    thought_evaluator: BaseThoughtEvaluator
    max_num_thoughts: int
    max_num_steps: int
    value_threshold: float
    thought_sorter: Optional[BaseThoughtSorter] = None
    do_sorting: bool = False  # True for DFS (Tree-of-Thoughts), False for DFSDT (ToolLLM)

    def _generate_thoughts(
        self,
        inputs: Dict[str, str],
        current_state: List[Tuple[AgentAction, str]],
        max_num_thoughts: Optional[int] = None,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> List[Union[AgentAction, AgentFinish]]:
        max_num_thoughts = self.max_num_thoughts if max_num_thoughts is None else max_num_thoughts
        return self.thought_generator.generate_thoughts(
            agent=self.agent,
            inputs=inputs,
            current_state=current_state,
            max_num_thoughts=max_num_thoughts,
            run_manager=run_manager.get_child(tag="generate_thoughts") if run_manager else None,
        )

    def _sort_thoughts(
        self,
        inputs: Dict[str, str],
        current_state: List[Tuple[AgentAction, str]],
        thoughts: List[Union[AgentAction, AgentFinish]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> List[Union[AgentAction, AgentFinish]]:
        assert self.thought_sorter is not None, "Sorting enabled, but thought sorter was not passed."
        return self.thought_sorter.sort_thoughts(
            thoughts=thoughts,
            inputs=inputs,
            current_state=current_state,
            run_manager=run_manager.get_child(tag="sort_thoughts") if run_manager else None,
        )

    def _evaluate_thought(
        self,
        inputs: Dict[str, str],
        current_state: List[Tuple[AgentAction, str]],
        thought: Union[AgentAction, AgentFinish],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> float:
        return self.thought_evaluator.evaluate_thought(
            inputs=inputs,
            current_state=current_state,
            next_thought=thought,
            run_manager=run_manager.get_child(tag="evaluate_thought") if run_manager else None,
        )

    def _dfs(
        self,
        inputs: Dict[str, str],
        cur_step: int,
        intermediate_steps: List[Tuple[AgentAction, str]],
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> AgentAction | AgentFinish:
        # stop when number of steps is more than T
        if cur_step > self.max_num_steps:
            return AgentFinish({"output": "Agent stopped due to iteration limit."}, "")

        # generate k possible next steps
        thoughts = self._generate_thoughts(inputs=inputs, current_state=intermediate_steps, run_manager=run_manager)

        # sort them
        if self.do_sorting:
            thoughts = self._sort_thoughts(
                inputs=inputs, current_state=intermediate_steps, thoughts=thoughts, run_manager=run_manager
            )

        for cur_thought in thoughts:
            # evaluate each thought
            cur_thought_value = self._evaluate_thought(
                inputs=inputs, current_state=intermediate_steps, thought=cur_thought, run_manager=run_manager
            )

            # proceed only with thoughts with value above a certain threshold
            if cur_thought_value > self.value_threshold:
                if isinstance(cur_thought, AgentFinish):
                    return cur_thought

                cur_result = self._perform_agent_action(
                    name_to_tool_map=name_to_tool_map,
                    color_mapping=color_mapping,
                    agent_action=cur_thought,
                    run_manager=run_manager,
                )

                return self._dfs(
                    inputs=inputs,
                    cur_step=cur_step + 1,
                    intermediate_steps=intermediate_steps + [(cur_result.action, cur_result.observation)],
                    name_to_tool_map=name_to_tool_map,
                    color_mapping=color_mapping,
                    run_manager=run_manager,
                )

        return AgentFinish({"output": "Agent stopped due to no promising thoughts available."}, "")

    def _run_strategy(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[AgentFinish, List[Tuple[AgentAction, str]]]:
        # TODO: process intermediate_steps (2nd output) correctly
        output = self._dfs(
            name_to_tool_map=name_to_tool_map,
            color_mapping=color_mapping,
            inputs=inputs,
            intermediate_steps=[],
            run_manager=run_manager,
            cur_step=0,
        )

        # TODO: ugly?
        assert isinstance(output, AgentFinish)
        return output, []
