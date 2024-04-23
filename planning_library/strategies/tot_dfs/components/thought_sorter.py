from __future__ import annotations
from textwrap import dedent
from typing import Dict, List, Optional, Tuple, Union, Any

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from itertools import combinations
from langchain_core.callbacks import AsyncCallbackManager, CallbackManager
from planning_library.components import BaseComponent, RunnableComponent
from planning_library.function_calling_parsers import (
    BaseFunctionCallingMultiActionParser,
    BaseFunctionCallingSingleActionParser,
    ParserRegistry,
)
from typing_extensions import TypedDict
from dataclasses import dataclass
from collections import defaultdict
from planning_library.utils import (
    format_thought,
)


@dataclass
class ThoughtSorterConfig:
    runnable: Optional[Runnable] = None

    prompt: Optional[ChatPromptTemplate] = None
    user_message: Optional[str] = None
    system_message: Optional[str] = None

    llm: Optional[BaseChatModel] = None

    parser: Optional[
        Union[
            BaseFunctionCallingSingleActionParser,
            BaseFunctionCallingMultiActionParser,
        ]
    ] = None
    parser_name: Optional[str] = None

    output_parser: Optional[BaseOutputParser[str]] = None


class ThoughtSorterInput(TypedDict):
    inputs: Dict[str, Any]
    thoughts: List[List[AgentAction] | AgentAction | AgentFinish]
    intermediate_steps: List[Tuple[AgentAction, str]]


class ThoughtSorterRunnableInput(TypedDict):
    intermediate_steps: List[Tuple[AgentAction, str]]
    thought1: List[BaseMessage]
    thought2: List[BaseMessage]


class ThoughtSorter(
    BaseComponent[
        ThoughtSorterInput, List[Union[List[AgentAction], AgentAction, AgentFinish]]
    ]
):
    """
    ToT+DFS component responsible for sorting the candidate thought on each DFS step.

    Follows the algorithm from ToolLLM repository (https://github.com/OpenBMB/ToolBench), specifically:
      1. Compares all pairs of candidate thoughts
      2. Computes final scores for each thought based on pairwise comparison
      3. Returns a list sorted by the scores (in descending order, the bigger the better).

    https://github.com/OpenBMB/ToolBench/blob/2937497244096960a532b21f66f663ed78e08588/toolbench/inference/LLM_rank/rank_candidate.py#L53
    """

    name = "Sort Thoughts"

    def __init__(
        self,
        runnable: Runnable[ThoughtSorterRunnableInput, str]
        | RunnableComponent[ThoughtSorterRunnableInput, str],
    ):
        if not isinstance(runnable, RunnableComponent):
            runnable = RunnableComponent(runnable)
        self.runnable = runnable

    @classmethod
    def _create_default_prompt(
        cls, system_message: Optional[str], user_message: str
    ) -> ChatPromptTemplate:
        if system_message is None:
            system_message = (
                "You are an advanced reasoning assistant that compares the "
                "steps suggested for solving complex reasoning tasks."
            )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    user_message,
                ),
                MessagesPlaceholder("intermediate_steps"),
                ("human", "Here is the first proposed next step"),
                MessagesPlaceholder("thought1"),
                ("human", "Here is the second proposed next step:"),
                MessagesPlaceholder("thought2"),
                (
                    "human",
                    dedent("""Your goal is to judge which of the proposed actions is more likely to lead to the success.

                         Take your time and comment your decision, 
                         but make sure to always output either 1 or 2, 
                         where 1 means 'the first proposed action is more likely to help' 
                         and 2 means 'the second proposed action is more likely to help'. 

                         ALWAYS use the following format and add the number in the end of your answer: [[number]].

                         Your verdict:
                         """),
                ),
            ]
        )

    def _compare_pairwise(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        thought1: List[AgentAction] | AgentAction | AgentFinish,
        thought2: List[AgentAction] | AgentAction | AgentFinish,
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> str:
        return self.runnable.invoke(
            {
                **inputs,  # type: ignore[typeddict-item]
                "intermediate_steps": intermediate_steps,
                "thought1": format_thought(thought1),
                "thought2": format_thought(thought2),
            },
            run_manager=run_manager,
            **kwargs,
        )

    async def _acompare_pairwise(
        self,
        inputs: Dict[str, Any],
        intermediate_steps: List[Tuple[AgentAction, str]],
        thought1: List[AgentAction] | AgentAction | AgentFinish,
        thought2: List[AgentAction] | AgentAction | AgentFinish,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> str:
        return await self.runnable.ainvoke(
            {
                **inputs,  # type: ignore[typeddict-item]
                "intermediate_steps": intermediate_steps,
                "thought1": format_thought(thought1),
                "thought2": format_thought(thought2),
            },
            run_manager=run_manager,
        )

    def invoke(
        self,
        inputs: ThoughtSorterInput,
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> List[Union[List[AgentAction], AgentAction, AgentFinish]]:
        scores: Dict[Union[List[AgentAction], AgentAction, AgentFinish], float] = (
            defaultdict(float)
        )
        for thought1, thought2 in [
            pair for pair in combinations(inputs["thoughts"], 2)
        ]:
            result = self._compare_pairwise(
                inputs=inputs["inputs"],
                intermediate_steps=inputs["intermediate_steps"],
                thought1=thought1,
                thought2=thought2,
                run_manager=run_manager,
                **kwargs,
            )

            if result == "1":
                scores[thought1] += 1
            elif result == "2":
                scores[thought2] += 1
            else:
                scores[thought1] += 0.5
                scores[thought2] += 0.5

        sorted_scores = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [key for key in sorted_scores]

    async def ainvoke(
        self,
        inputs: ThoughtSorterInput,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> List[Union[List[AgentAction], AgentAction, AgentFinish]]:
        scores: Dict[Union[List[AgentAction], AgentAction, AgentFinish], float] = (
            defaultdict(float)
        )
        for thought1, thought2 in [
            pair for pair in combinations(inputs["thoughts"], 2)
        ]:
            result = await self._acompare_pairwise(
                inputs=inputs["inputs"],
                intermediate_steps=inputs["intermediate_steps"],
                thought1=thought1,
                thought2=thought2,
                run_manager=run_manager,
            )

            if result == "1":
                scores[thought1] += 1
            elif result == "2":
                scores[thought2] += 1
            else:
                scores[thought1] += 0.5
                scores[thought2] += 0.5

        sorted_scores = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [key for key in sorted_scores]

    @classmethod
    def create_from_config(cls, config: ThoughtSorterConfig) -> ThoughtSorter:
        def _preprocess_input(
            inputs: ThoughtSorterInput,
        ) -> Dict:
            # TODO: figure out typing here
            nonlocal config
            if config.parser is None:
                assert config.parser_name is not None
                parser = ParserRegistry.get_parser(config.parser_name)
            else:
                parser = config.parser

            intermediate_steps = parser.format_inputs(inputs)["agent_scratchpad"]
            return {
                **inputs["inputs"],
                "thoughts": inputs["thoughts"],
                "intermediate_steps": intermediate_steps,
            }

        if config.runnable is not None:
            return cls(config.runnable)

        if config.llm is None:
            raise ValueError("`llm` must be provided when `runnable` is None.")

        if config.output_parser is None:
            raise ValueError(
                "Default output parser for thought sorter is not implemented yet.`output_parser` must be provided when `runnable` is None."
            )

        prompt = cls._process_prompt(
            prompt=config.prompt,
            user_message=config.user_message,
            system_message=config.system_message,
        )

        sorter_runnable = RunnableComponent.create_from_steps(
            prompt=prompt, llm=config.llm, output_parser=config.output_parser
        )

        sorter_runnable.add_input_preprocessing(_preprocess_input)

        return cls(sorter_runnable)
