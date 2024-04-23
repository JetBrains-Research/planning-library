from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from textwrap import dedent
from typing import Optional

from planning_library.action_executors import BaseActionExecutor, DefaultActionExecutor
from planning_library.components import RunnableComponent
from planning_library.components.agent_component import AgentFactory
from planning_library.strategies import SimpleStrategy
from typing import Dict, Any, Tuple, List
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import Runnable
from typing_extensions import TypedDict
from typing import Union, Sequence
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from planning_library.function_calling_parsers import (
    BaseFunctionCallingSingleActionParser,
    BaseFunctionCallingMultiActionParser,
)
from dataclasses import dataclass


class ADaPTExecutorInput(TypedDict):
    inputs: Dict[str, Any]


class ADaPTExecutorOutput(TypedDict):
    is_completed: bool
    agent_outcome: AgentFinish
    intermediate_steps: List[Tuple[AgentAction, str]]


@dataclass
class ADaPTExecutorConfig:
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
    executor_parser: Optional[
        Union[
            BaseFunctionCallingSingleActionParser,
            BaseFunctionCallingMultiActionParser,
        ]
    ] = None
    executor_parser_name: Optional[str] = None


class ADaPTExecutor(RunnableComponent[ADaPTExecutorInput, ADaPTExecutorOutput]):
    name = "Executor"

    def __init__(self, runnable: Runnable[ADaPTExecutorInput, ADaPTExecutorOutput]):
        super().__init__(runnable)
        self.add_output_preprocessing(ADaPTExecutor._process_outputs)  # type: ignore[arg-type]

    @classmethod
    def _create_default_prompt(
        cls, system_message: Optional[str], user_message: str
    ) -> ChatPromptTemplate:
        if system_message is None:
            system_message = "You are an advanced reasoning agent."

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
                        When you are sure that you have successfully completed the task,
                         avoid calling tools and make sure to include the words 'task completed' in your output.
                        Do not write 'task completed' if the task has not been completed, for instance,
                         if you think something went wrong and you would like to try again.
                        """),
                ),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

    @staticmethod
    def _process_outputs(outputs: Dict[str, Any]) -> ADaPTExecutorOutput:
        outcome = AgentFinish(
            return_values={
                key: value[0]
                for key, value in outputs.items()
                if isinstance(key, list)
                and key not in ["finish_log", "intermediate_steps"]
            },
            log=outputs["finish_log"][0],
        )

        return {
            "is_completed": "task completed" in outcome.log.lower(),
            "agent_outcome": outcome,
            "intermediate_steps": outputs["intermediate_steps"][0],
        }

    @classmethod
    def create_simple_strategy(
        cls,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        action_executor: Optional[BaseActionExecutor] = None,
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
        return_intermediate_steps: bool = False,
        return_finish_log: bool = False,
        max_iterations: int = 15,
        verbose: bool = True,
        **kwargs,
    ) -> "ADaPTExecutor":
        prompt = cls._process_prompt(
            prompt=prompt, user_message=user_message, system_message=system_message
        )

        agent = AgentFactory.create_agent(
            llm=llm, tools=tools, prompt=prompt, parser=parser, parser_name=parser_name
        )

        strategy = SimpleStrategy.create(
            tools=tools,
            action_executor=action_executor
            if action_executor is not None
            else DefaultActionExecutor(tools),
            agent=agent,
            return_intermediate_steps=return_intermediate_steps,
            return_finish_log=return_finish_log,
            max_iterations=max_iterations,
            verbose=verbose,
        )
        return cls(runnable=strategy)  # type: ignore[arg-type]
