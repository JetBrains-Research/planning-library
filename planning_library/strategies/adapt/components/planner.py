from __future__ import annotations

from langchain_core.callbacks import CallbackManager, AsyncCallbackManager
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from textwrap import dedent
from typing import Optional, Literal

from planning_library.action_executors import DefaultActionExecutor
from planning_library.strategies.adapt.utils import get_adapt_planner_tools
from planning_library.strategies.adapt.utils.planner_tools import BaseADaPTPlannerTool
from planning_library.components import BaseComponent, RunnableComponent
from planning_library.components.agent_component import AgentFactory
from planning_library.strategies import SimpleStrategy
from typing import Dict, Any, Tuple, List
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import Runnable
from typing_extensions import TypedDict
from typing import Union, Sequence
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from planning_library.utils import format_thought
from langchain.agents.agent import RunnableAgent, RunnableMultiActionAgent
from langchain_core.runnables import RunnableLambda
from planning_library.function_calling_parsers import (
    BaseFunctionCallingSingleActionParser,
    BaseFunctionCallingMultiActionParser,
    ParserRegistry,
)
from dataclasses import dataclass


class ADaPTPlannerInput(TypedDict):
    inputs: Dict[str, Any]
    executor_agent_outcome: AgentFinish
    executor_intermediate_steps: List[Tuple[AgentAction, str]]


class ADaPTPlannerOutput(TypedDict):
    subtasks: List[str]
    aggregation_mode: Literal["and", "or"]


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


class ADaPTPlanner(BaseComponent[ADaPTPlannerInput, ADaPTPlannerOutput]):
    name = "Planner"

    def __init__(
        self,
        runnable: Runnable | RunnableComponent[ADaPTPlannerInput, Any],
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
        cls, system_message: Optional[str], user_message: str, mode: str = "agent"
    ) -> ChatPromptTemplate:
        if mode == "agent":
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
                        dedent("""
                            Here is the final outcome of the original task: {executor_agent_outcome}.
    
                            Below, you will find the intermediate steps that were taken in order to achieve this outcome.
                            """),
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
    
                            You are given access to a set of tools to help with the plan construction. When you are done, simply refrain from calling any tools.                          
                            """),
                    ),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )
        raise NotImplementedError("Currently, only agentic planner is supported.")

    def invoke(
        self,
        inputs: ADaPTPlannerInput,
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> ADaPTPlannerOutput:
        if self.mode == "agent":
            assert (
                self.tools is not None and len(self.tools) > 0
            ), "Tools have to be defined for agentic mode."
            plan = self.tools[0].plan
            plan.clear()
            _ = self.runnable.invoke(
                inputs, run_manager=run_manager, run_name=self.name
            )
            return {
                "subtasks": plan.subtasks,
                "aggregation_mode": plan.aggregation_mode,
            }

        raise NotImplementedError("Currently, only agentic planner is supported.")

    async def ainvoke(
        self,
        inputs: ADaPTPlannerInput,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> ADaPTPlannerOutput:
        if self.mode == "agent":
            assert (
                self.tools is not None and len(self.tools) > 0
            ), "Tools have to be defined for agentic mode."
            plan = self.tools[0].plan
            plan.clear()
            _ = await self.runnable.ainvoke(
                inputs, run_manager=run_manager, run_name=self.name
            )
            return {
                "subtasks": plan.subtasks,
                "aggregation_mode": plan.aggregation_mode,
            }

        raise NotImplementedError("Currently, only agentic planner is supported.")

    @classmethod
    def create_planner_agent(
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
        executor_parser: Optional[
            Union[
                BaseFunctionCallingSingleActionParser,
                BaseFunctionCallingMultiActionParser,
            ]
        ] = None,
        executor_parser_name: Optional[str] = None,
    ) -> Union[RunnableAgent, RunnableMultiActionAgent]:
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
                "intermediate_steps": inputs["intermediate_steps"],
                "executor_agent_outcome": format_thought(
                    inputs["executor_agent_outcome"]
                ),
                "executor_intermediate_steps": executor_intermediate_steps,
            }

        prompt = cls._process_prompt(
            prompt=prompt, user_message=user_message, system_message=system_message
        )

        agent = AgentFactory.create_agent(
            llm=llm, tools=tools, prompt=prompt, parser=parser, parser_name=parser_name
        )
        agent.runnable = RunnableLambda(_preprocess_input) | agent.runnable  # type: ignore[assignment]
        return agent

    @classmethod
    def create_simple_strategy(
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
        tools = tools if tools is not None else get_adapt_planner_tools()  # type: ignore[assignment]

        agent = cls.create_planner_agent(
            llm=llm,
            tools=tools,  # type: ignore[arg-type]
            prompt=prompt,
            user_message=user_message,
            system_message=system_message,
            parser=parser,
            parser_name=parser_name,
            executor_parser=executor_parser,
            executor_parser_name=executor_parser_name,
        )

        strategy = SimpleStrategy.create(
            tools=tools,  # type: ignore[arg-type]
            action_executor=DefaultActionExecutor(tools),  # type: ignore[arg-type]
            agent=agent,
            return_intermediate_steps=return_intermediate_steps,
            return_finish_log=return_finish_log,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        return cls(runnable=strategy, tools=tools)
