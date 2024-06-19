from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain.agents import BaseMultiActionAgent, BaseSingleActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import AsyncCallbackManager, CallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict

from planning_library.components import AgentComponent, BaseComponent
from planning_library.function_calling_parsers import (
    BaseFunctionCallingMultiActionParser,
    BaseFunctionCallingSingleActionParser,
)
from planning_library.utils import (
    format_thoughts,
)


@dataclass
class ThoughtGeneratorConfig:
    tools: Sequence[BaseTool]
    max_num_thoughts: int

    prompt: Optional[ChatPromptTemplate] = None
    user_message: Optional[str] = None
    system_message: Optional[str] = None

    agent: Optional[BaseMultiActionAgent | BaseSingleActionAgent] = None

    llm: Optional[BaseChatModel] = None

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


class ThoughtGenerator(BaseComponent[ThoughtGeneratorInput, List[Union[List[AgentAction], AgentAction, AgentFinish]]]):
    name = "Generate Thoughts"

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
    def _create_default_prompt(cls, system_message: Optional[str], user_message: str, **kwargs) -> ChatPromptTemplate:
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
                (
                    "human",
                    "You might have already made some suggestions for the current state - if you did, you will find them below.",
                ),
                MessagesPlaceholder("previous_thoughts"),
                (
                    "human",
                    "Please, remember to suggest exactly ONE (1) tool call, no more and no less, different from your previous suggestions.",
                ),
            ]
        )

    def invoke(
        self,
        inputs: ThoughtGeneratorInput,
        run_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> List[List[AgentAction] | AgentAction | AgentFinish]:
        results: List[List[AgentAction] | AgentAction | AgentFinish] = []
        for _ in range(self.max_num_thoughts):
            cur_result = self.agent.invoke(
                {**inputs, "previous_thoughts": results},
                run_manager=run_manager,
                **kwargs,
            )
            # TODO: how to fix mypy warning properly here?
            results.append(cur_result)  # type: ignore[arg-type]

        return results

    async def ainvoke(
        self,
        inputs: ThoughtGeneratorInput,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> List[List[AgentAction] | AgentAction | AgentFinish]:
        results: List[List[AgentAction] | AgentAction | AgentFinish] = []
        for _ in range(self.max_num_thoughts):
            cur_result = await self.agent.ainvoke(
                {**inputs, "previous_thoughts": results},
                run_manager=run_manager,
                **kwargs,
            )
            # TODO: how to fix mypy warning properly here?
            results.append(cur_result)  # type: ignore[arg-type]

        return results

    @classmethod
    def create_from_config(cls, config: ThoughtGeneratorConfig) -> ThoughtGenerator:
        if config.agent is not None:
            return ThoughtGenerator(agent=config.agent, max_num_thoughts=config.max_num_thoughts)

        if config.llm is None:
            raise ValueError("`llm` must be provided when `agent` is None.")

        if config.tools is None:
            raise ValueError("`tools` must be provided when `agent` is None.")

        return cls.create(
            llm=config.llm,
            tools=config.tools,
            prompt=config.prompt,
            user_message=config.user_message,
            system_message=config.system_message,
            parser=config.parser,
            parser_name=config.parser_name,
            max_num_thoughts=config.max_num_thoughts,
        )

    @classmethod
    def create(
        cls,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        max_num_thoughts: int,
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
    ) -> ThoughtGenerator:
        prompt = cls._process_prompt(prompt=prompt, user_message=user_message, system_message=system_message)

        agent: AgentComponent = AgentComponent.create(
            llm=llm,
            tools=tools,
            prompt=prompt,
            parser=parser,
            parser_name=parser_name,
        )

        agent.add_input_preprocessing(
            preprocess=lambda inputs: {
                **{key: value for key, value in inputs.items() if key != "previous_thoughts"},
                "previous_thoughts": format_thoughts(inputs["previous_thoughts"]),
            }
        )

        return ThoughtGenerator(agent=agent, max_num_thoughts=max_num_thoughts)
