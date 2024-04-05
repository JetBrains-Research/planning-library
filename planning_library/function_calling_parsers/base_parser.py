from abc import ABC, abstractmethod
from typing import Any, List, Tuple
from langchain.agents.agent import AgentOutputParser, MultiActionAgentOutputParser
from langchain_core.agents import AgentAction
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentInputs(TypedDict):
    intermediate_steps: List[Tuple[AgentAction, str]]


class ProcessedAgentInputs(TypedDict):
    agent_scratchpad: List[BaseMessage]


class BaseFunctionCallingParser(ABC):
    name: str

    @abstractmethod
    def format_inputs(self, inputs: AgentInputs) -> ProcessedAgentInputs: ...

    @abstractmethod
    def prepare_tool(self, tool: BaseTool) -> Any: ...


class BaseFunctionCallingSingleActionParser(BaseFunctionCallingParser, ABC):
    output_parser: AgentOutputParser


class BaseFunctionCallingMultiActionParser(BaseFunctionCallingParser, ABC):
    output_parser: MultiActionAgentOutputParser
