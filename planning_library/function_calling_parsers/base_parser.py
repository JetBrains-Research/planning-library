from abc import ABC, abstractmethod
from typing import List, Tuple, Sequence, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain.agents.agent import AgentOutputParser, MultiActionAgentOutputParser
from langchain_core.agents import AgentAction
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentInputs(TypedDict):
    inputs: Dict[str, Any]
    intermediate_steps: List[Tuple[AgentAction, str]]


class ProcessedAgentInputs(TypedDict):
    inputs: Dict[str, Any]
    agent_scratchpad: List[BaseMessage]


class BaseFunctionCallingParser(ABC):
    name: str

    @abstractmethod
    def prepare_llm(
        self, llm: BaseChatModel, tools: Sequence[BaseTool]
    ) -> Runnable: ...

    @abstractmethod
    def format_inputs(self, inputs: AgentInputs) -> ProcessedAgentInputs: ...


class BaseFunctionCallingSingleActionParser(BaseFunctionCallingParser, ABC):
    output_parser: AgentOutputParser


class BaseFunctionCallingMultiActionParser(BaseFunctionCallingParser, ABC):
    output_parser: MultiActionAgentOutputParser
