from abc import ABC, abstractmethod
from typing import Generic, Any, List, Tuple, TypeVar, Union, Optional
from langchain.agents.agent import AgentOutputParser, MultiActionAgentOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.outputs import Generation
from typing_extensions import TypedDict


class AgentInputs(TypedDict):
    intermediate_steps: List[Tuple[AgentAction, str]]


class ProcessedAgentInputs(TypedDict):
    agent_scratchpad: List[BaseMessage]


Inputs = TypeVar("Inputs", bound=AgentInputs)
ProcessedInputs = TypeVar("ProcessedInputs", bound=ProcessedAgentInputs)


class BaseFunctionCallingParser(Generic[Inputs, ProcessedInputs], ABC):
    name: str

    @abstractmethod
    def format_inputs(self, inputs: Inputs) -> ProcessedInputs: ...

    @abstractmethod
    def prepare_tool(self, tool: BaseTool) -> Any: ...


class BaseFunctionCallingSingleActionParser(BaseFunctionCallingParser, ABC):
    output_parser: Optional[AgentOutputParser] = None

    def parse_outputs(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[AgentAction, AgentFinish]:
        if self.output_parser is not None:
            return self.output_parser.parse_result(result=result, partial=partial)
        raise NotImplementedError(
            "By default, this class uses output_parser. "
            "Please, either provide output_parser or redefine parse_outputs method explicitly."
        )


class BaseFunctionCallingMultiActionParser(BaseFunctionCallingParser, ABC):
    output_parser: Optional[MultiActionAgentOutputParser] = None

    def parse_outputs(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[AgentAction], AgentFinish]:
        if self.output_parser is not None:
            return self.output_parser.parse_result(result=result, partial=partial)
        raise NotImplementedError(
            "By default, this class uses output_parser. "
            "Please, either provide output_parser or redefine parse_outputs method explicitly."
        )
