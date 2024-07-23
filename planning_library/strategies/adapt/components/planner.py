from __future__ import annotations

from textwrap import dedent
from typing import Any, Dict, Optional, Union

from langchain_core.callbacks import AsyncCallbackManager, CallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from planning_library.components import BaseComponent, RunnableComponent
from planning_library.function_calling_parsers import (
    BaseFunctionCallingMultiActionParser,
    BaseFunctionCallingSingleActionParser,
    ParserRegistry,
)
from planning_library.strategies.adapt.utils import ADaPTPlannerInput, ADaPTPlannerOutput, SimplePlannerOutputParser
from planning_library.utils import format_thought


class ADaPTPlanner(BaseComponent[ADaPTPlannerInput, ADaPTPlannerOutput]):
    name = "Planner"

    def __init__(
        self,
        runnable: Runnable[ADaPTPlannerInput, ADaPTPlannerOutput]
        | RunnableComponent[ADaPTPlannerInput, ADaPTPlannerOutput],
    ):
        if not isinstance(runnable, RunnableComponent):
            runnable = RunnableComponent(runnable)
        self.runnable = runnable

    @classmethod
    def _create_default_prompt(
        cls,
        system_message: Optional[str],
        user_message: str,
        **kwargs,
    ) -> ChatPromptTemplate:
        if system_message is None:
            system_message = (
                "You are an advanced Planning agent that decomposes auxiliary tasks into step-by-step "
                "plans for another advanced agent, Executor."
            )

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    user_message,
                ),
                (
                    "human",
                    "Here is the final outcome of the original task:",
                ),
                MessagesPlaceholder("executor_agent_outcome"),
                (
                    "human",
                    "Below, you will find the intermediate steps that were taken in order to achieve this outcome.",
                ),
                MessagesPlaceholder("executor_intermediate_steps"),
                (
                    "human",
                    dedent("""
                            The trial above was unsuccessful. Your goal is to construct a step-by-step plan to successfully solve the original task.
                            Plan is a list of subtasks with the logic of how the subtasks' results should be aggregated.

                            You are only allowed to devise plans either with "and" aggregation logic or "or" aggregation logic.
                            For "and" logic, the original task will only be considered solved if ALL subtasks are successfully solved.
                            For "or" logic, the original task will only be considered solved if ANY of the subtasks are successfully solved.
                            In both cases, subtasks will be executed sequentially. 
                            For "and" logic, the execution will stop at the first failure.
                            For "or" logic, the execution will stop at the first success.

                            Each subtask will be passed to an executor agent separately. Make sure to make all the instructions self-sufficient (containing all the necessary information, observations, etc.), yet concise.

                            Use JSON format to output a plan. It should contain keys 'subtasks' (list of subtasks, where each subtask is defined by a simple natural-language instruction, a single string) and 'aggregation_mode' (logic for uniting the subtasks results), like that:
                            ```
                            {{"subtasks": [
                                          "<subtask1 instruction>", 
                                          "<subtask2 instruction>", 
                                          ..., 
                                          "<subtaskn instruction>"],
                             "aggregation_mode": "<aggregation mode>"}}
                            ```    
                            """),
                ),
            ]
        )

    def invoke(
        self,
        inputs: ADaPTPlannerInput,
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> ADaPTPlannerOutput:
        return self.runnable.invoke(inputs, run_manager=run_manager, run_name=self.name)

    async def ainvoke(
        self,
        inputs: ADaPTPlannerInput,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> ADaPTPlannerOutput:
        return await self.runnable.ainvoke(inputs, run_manager=run_manager, run_name=self.name)

    @classmethod
    def create(
        cls,
        llm: BaseChatModel,
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        executor_parser: Optional[
            Union[
                BaseFunctionCallingSingleActionParser,
                BaseFunctionCallingMultiActionParser,
            ]
        ] = None,
        executor_parser_name: Optional[str] = None,
        output_parser=None,
        **kwargs,
    ) -> "ADaPTPlanner":
        def _preprocess_input(
            inputs: Dict[str, Any],
        ) -> Dict[str, Any]:
            # TODO: figure out typing here
            nonlocal executor_parser, executor_parser_name
            if executor_parser is None:
                assert executor_parser_name is not None
                executor_parser = ParserRegistry.get_parser(executor_parser_name)

            executor_intermediate_steps = executor_parser.format_inputs(
                {
                    "inputs": inputs["inputs"],
                    "intermediate_steps": inputs["executor_intermediate_steps"],
                }
            )["agent_scratchpad"]

            return {
                **inputs["inputs"],
                "executor_agent_outcome": format_thought(inputs["executor_agent_outcome"]),
                "executor_intermediate_steps": executor_intermediate_steps,
            }

        prompt = cls._process_prompt(
            prompt=prompt,
            user_message=user_message,
            system_message=system_message,
            mode="simple",
        )

        if output_parser is None:
            output_parser = SimplePlannerOutputParser()

        runnable = RunnableComponent.create_from_steps(prompt=prompt, llm=llm, output_parser=output_parser)
        runnable.add_input_preprocessing(_preprocess_input)
        return cls(runnable=runnable)
