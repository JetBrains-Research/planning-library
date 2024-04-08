from typing import List, Tuple, Sequence
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_core.messages import BaseMessage
from langchain_core.agents import AgentAction
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from planning_library.function_calling_parsers.base_parser import (
    BaseFunctionCallingMultiActionParser,
    AgentInputs,
    ProcessedAgentInputs,
)
from planning_library.function_calling_parsers.parser_registry import ParserRegistry


@ParserRegistry.register
class OpenAIToolsParser(BaseFunctionCallingMultiActionParser):
    name: str = "openai-tools"
    output_parser = OpenAIToolsAgentOutputParser()

    @staticmethod
    def _format_intermediate_steps(
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> List[BaseMessage]:
        return format_to_openai_tool_messages(intermediate_steps)

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

    def prepare_llm(self, llm: BaseChatModel, tools: Sequence[BaseTool]) -> Runnable:
        return llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])
