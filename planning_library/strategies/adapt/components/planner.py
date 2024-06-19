from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, Optional, Sequence, Union

from langchain.agents.agent import RunnableAgent, RunnableMultiActionAgent
from langchain_core.callbacks import AsyncCallbackManager, CallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool

from planning_library.action_executors import LangchainActionExecutor
from planning_library.components import BaseComponent, RunnableComponent
from planning_library.components.agent_component import AgentFactory
from planning_library.function_calling_parsers import (
    BaseFunctionCallingMultiActionParser,
    BaseFunctionCallingSingleActionParser,
    ParserRegistry,
)
from planning_library.strategies import SimpleStrategy
from planning_library.strategies.adapt.utils import (
    ADaPTPlannerInput,
    ADaPTPlannerOutput,
    SimplePlannerOutputParser,
    get_adapt_planner_tools,
)
from planning_library.strategies.adapt.utils.planner_tools import BaseADaPTPlannerTool
from planning_library.utils import format_thought


@dataclass
class ADaPTPlannerConfig:
    runnable: Optional[Runnable] = None
    llm: Optional[BaseChatModel] = None
    tools: Optional[Sequence[BaseTool]] = None
    prompt: Optional[ChatPromptTemplate] = None
    user_message: Optional[str] = None
    system_message: Optional[str] = None
    parser: Optional[
        Union[
            BaseFunctionCallingSingleActionParser,
            BaseFunctionCallingMultiActionParser,
        ]
    ] = None
    parser_name: Optional[str] = None
    output_parser: Optional[BaseOutputParser[ADaPTPlannerOutput]] = None
    mode: str = "agent"


class ADaPTPlanner(BaseComponent[ADaPTPlannerInput, ADaPTPlannerOutput]):
    name = "Planner"

    def __init__(
        self,
        runnable: Runnable
        | RunnableComponent[ADaPTPlannerInput, Any]
        | RunnableComponent[ADaPTPlannerInput, ADaPTPlannerOutput],
        tools: Optional[Sequence[BaseADaPTPlannerTool]] = None,
        mode: str = "agent",
    ):
        if not isinstance(runnable, RunnableComponent):
            runnable = RunnableComponent(runnable)
        self.runnable = runnable
        self.tools = tools
        self.mode = mode

    @classmethod
    def _create_default_prompt(
        cls,
        system_message: Optional[str],
        user_message: str,
        mode: str = "agent",
        **kwargs,
    ) -> ChatPromptTemplate:
        if system_message is None:
            system_message = (
                "You are an advanced Planning agent that decomposes auxiliary tasks into step-by-step "
                "plans for another advanced agent, Executor."
            )

        base_messages = [
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
        ]

        if mode == "agent":
            base_messages += [
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
    
                            You are given access to a set of tools to help with the plan construction. ALWAYS use tools, refrain from using tools only when you are done.                          
                            """),
                ),
                MessagesPlaceholder("agent_scratchpad"),
            ]

        elif mode == "simple":
            base_messages += [
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
        else:
            raise NotImplementedError(f"Unsupported mode {mode}.")

        return ChatPromptTemplate.from_messages(
            base_messages  # type: ignore[arg-type]
        )

    def invoke(
        self,
        inputs: ADaPTPlannerInput,
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> ADaPTPlannerOutput:
        if self.mode == "agent":
            assert self.tools is not None and len(self.tools) > 0, "Tools have to be defined for agentic mode."
            plan = self.tools[0].plan
            plan.clear()
            _ = self.runnable.invoke(inputs, run_manager=run_manager, run_name=self.name)
            return {
                "subtasks": plan.subtasks,
                "aggregation_mode": plan.aggregation_mode,
            }
        if self.mode == "simple":
            return self.runnable.invoke(inputs, run_manager=run_manager, run_name=self.name)

        raise NotImplementedError(
            "Currently, only `agent` (with tools) and `simple` (a single call to llm) modes for the planner are supported."
        )

    async def ainvoke(
        self,
        inputs: ADaPTPlannerInput,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> ADaPTPlannerOutput:
        if self.mode == "agent":
            assert self.tools is not None and len(self.tools) > 0, "Tools have to be defined for agentic mode."
            plan = self.tools[0].plan
            plan.clear()
            _ = await self.runnable.ainvoke(inputs, run_manager=run_manager, run_name=self.name)
            return {
                "subtasks": plan.subtasks,
                "aggregation_mode": plan.aggregation_mode,
            }
        if self.mode == "simple":
            return await self.runnable.ainvoke(inputs, run_manager=run_manager, run_name=self.name)

        raise NotImplementedError("Currently, only agentic planner is supported.")

    @classmethod
    def _create_agent(
        cls,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        parser: Optional[
            Union[
                BaseFunctionCallingSingleActionParser,
                BaseFunctionCallingMultiActionParser,
            ]
        ] = None,
        parser_name: Optional[str] = None,
    ) -> Union[RunnableAgent, RunnableMultiActionAgent]:
        prompt = cls._process_prompt(prompt=prompt, user_message=user_message, system_message=system_message)

        return AgentFactory.create_agent(llm=llm, tools=tools, prompt=prompt, parser=parser, parser_name=parser_name)

    @classmethod
    def create_agent_planner(
        cls,
        llm: BaseChatModel,
        tools: Optional[Sequence[BaseADaPTPlannerTool]] = None,
        prompt: Optional[ChatPromptTemplate] = None,
        user_message: Optional[str] = None,
        system_message: Optional[str] = None,
        parser: Optional[
            Union[
                BaseFunctionCallingSingleActionParser,
                BaseFunctionCallingMultiActionParser,
            ]
        ] = None,
        parser_name: Optional[str] = None,
        executor_parser: Optional[
            Union[
                BaseFunctionCallingSingleActionParser,
                BaseFunctionCallingMultiActionParser,
            ]
        ] = None,
        executor_parser_name: Optional[str] = None,
        return_intermediate_steps: bool = False,
        return_finish_log: bool = False,
        max_iterations: int = 15,
        verbose: bool = True,
        **kwargs,
    ) -> "ADaPTPlanner":
        def _preprocess_input(
            inputs: ADaPTPlannerInput,
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

        tools = tools if tools is not None else get_adapt_planner_tools()  # type: ignore[assignment]

        agent = cls._create_agent(
            llm=llm,
            tools=tools,  # type: ignore[arg-type]
            prompt=prompt,
            user_message=user_message,
            system_message=system_message,
            parser=parser,
            parser_name=parser_name,
        )

        strategy = SimpleStrategy.create(
            tools=tools,  # type: ignore[arg-type]
            action_executor=LangchainActionExecutor(tools),  # type: ignore[arg-type]
            agent=agent,
            return_intermediate_steps=return_intermediate_steps,
            return_finish_log=return_finish_log,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        runnable = RunnableLambda(_preprocess_input) | strategy
        planner = cls(runnable=runnable, tools=tools)
        return planner

    @classmethod
    def create_simple_planner(
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
        return cls(runnable=runnable, mode="simple")
