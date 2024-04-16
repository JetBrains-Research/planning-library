from __future__ import annotations
from typing import Tuple, Dict, Any, List, Union, Optional, Sequence

from langchain_core.callbacks import CallbackManager, AsyncCallbackManager
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from typing_extensions import TypedDict
from planning_library.components import BaseComponent, AgentComponent
from planning_library.function_calling_parsers import (
    BaseFunctionCallingSingleActionParser,
    BaseFunctionCallingMultiActionParser,
)
from dataclasses import dataclass


@dataclass
class ThoughtGeneratorConfig:
    max_num_thought: int
    prompt: Optional[ChatPromptTemplate] = None
    user_message: Optional[str] = None
    system_message: Optional[str] = None
    agent: Optional[BaseMultiActionAgent | BaseSingleActionAgent] = None
    llm: Optional[BaseChatModel] = None
    tools: Optional[Sequence[BaseTool]] = None
    parser: Optional[
        Union[
            BaseFunctionCallingSingleActionParser,
            BaseFunctionCallingMultiActionParser,
        ]
    ] = None
    parser_name: Optional[str] = None


class ThoughtGeneratorInput(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[Tuple[AgentAction, str]]


class ThoughtGeneratorAgentInput(ThoughtGeneratorInput):
    previous_thoughts: List[List[AgentAction] | AgentAction | AgentFinish]


class ThoughtGenerator(
    BaseComponent[
        ThoughtGeneratorInput, List[Union[List[AgentAction], AgentAction, AgentFinish]]
    ]
):
    required_prompt_input_vars = set(ThoughtGeneratorInput.__annotations__) - {
        "inputs",
        "intermediate_steps",
    } | {"agent_scratchpad"}

    def __init__(
        self,
        agent: AgentComponent | BaseMultiActionAgent | BaseSingleActionAgent,
        max_num_thoughts: int,
    ):
        self.agent: AgentComponent[ThoughtGeneratorAgentInput] = (
            AgentComponent(agent) if not isinstance(agent, AgentComponent) else agent
        )
        self.max_num_thoughts = max_num_thoughts

    @classmethod
    def _create_default_prompt(
        cls, system_message: Optional[str], user_message: str
    ) -> ChatPromptTemplate:
        if system_message is None:
            system_message = "You are an advanced reasoning agent that can improve based on self-reflection."

        return ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                (
                    "human",
                    user_message,
                ),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

    def invoke(
        self,
        inputs: ThoughtGeneratorInput,
        run_manager: Optional[CallbackManager] = None,
    ) -> List[List[AgentAction] | AgentAction | AgentFinish]:
        results: List[List[AgentAction] | AgentAction | AgentFinish] = []
        for _ in range(self.max_num_thoughts):
            cur_result = self.agent.invoke(
                {**inputs, "previous_thoughts": results},
                run_manager=run_manager,
            )
            # TODO: how to fix mypy warning properly here?
            results.append(cur_result)  # type: ignore[arg-type]

        return results

    async def ainvoke(
        self,
        inputs: ThoughtGeneratorInput,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> List[List[AgentAction] | AgentAction | AgentFinish]:
        results: List[List[AgentAction] | AgentAction | AgentFinish] = []
        for _ in range(self.max_num_thoughts):
            cur_result = await self.agent.ainvoke(
                {**inputs, "previous_thoughts": results},
                run_manager=run_manager,
            )
            # TODO: how to fix mypy warning properly here?
            results.append(cur_result)  # type: ignore[arg-type]

        return results

    @classmethod
    def create_from_config(cls, config: ThoughtGeneratorConfig) -> ThoughtGenerator:
        if config.agent is not None:
            return ThoughtGenerator(
                agent=config.agent, max_num_thoughts=config.max_num_thought
            )

        if config.llm is None:
            raise ValueError("`llm` must be provided when `agent` is None.")

        if config.tools is None:
            raise ValueError("`tools` must be provided when `agent` is None.")

        agent: AgentComponent = AgentComponent.create(
            llm=config.llm,
            tools=config.tools,
            prompt=config.prompt,
            user_message=config.user_message,
            system_message=config.system_message,
            parser=config.parser,
            parser_name=config.parser_name,
        )
        return ThoughtGenerator(agent=agent, max_num_thoughts=config.max_num_thought)
