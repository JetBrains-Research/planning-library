from typing import Any, List, Tuple
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function

from planning_library.function_calling_parsers.base_parser import (
    BaseFunctionCallingSingleActionParser,
    ProcessedAgentInputs,
    AgentInputs,
)
from planning_library.function_calling_parsers.parser_registry import ParserRegistry


@ParserRegistry.register
class OpenAIFunctionsParser(BaseFunctionCallingSingleActionParser):
    name: str = "openai-functions"
    output_parser = OpenAIFunctionsAgentOutputParser()

    @staticmethod
    def _format_intermediate_steps(
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> List[BaseMessage]:
        return format_to_openai_function_messages(intermediate_steps)

    def format_inputs(
        self,
        inputs: AgentInputs,
    ) -> ProcessedAgentInputs:
        return {
            **inputs,  # type: ignore[typeddict-unknown-key]
            "agent_scratchpad": self._format_intermediate_steps(
                inputs["intermediate_steps"]
            ),
        }

    def prepare_tool(self, tool: BaseTool) -> Any:
        return convert_to_openai_function(tool)
