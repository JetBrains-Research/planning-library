from langchain.agents import BaseSingleActionAgent
from langchain_core.callbacks import Callbacks
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from typing import Sequence, Any, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.utils.function_calling import convert_to_openai_function


class PlanningLibBaseSingleActionAgent(BaseSingleActionAgent, ABC):
    _llm: Runnable
    tools: Sequence[BaseTool]
    prompt: ChatPromptTemplate
    output_parser: BaseOutputParser[Union[AgentAction, AgentFinish]]
    _runnable: Optional[Runnable]

    @property
    def llm(self):
        return self._llm

    @llm.setter
    def llm(self, llm: BaseChatModel):
        self._llm = llm.bind(
            tools=[self.convert_tool(tool) for tool in self.tools]
            if self.convert_tool is not None
            else self.tools
        )

    @staticmethod
    @abstractmethod
    def format_intermediate_steps(
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> List[BaseMessage]: ...

    @staticmethod
    @abstractmethod
    def convert_tool(tool: BaseTool) -> Any: ...

    def to_runnable(self) -> Runnable:
        return (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: self.format_intermediate_steps(
                    x["intermediate_steps"]
                )
            )
            | self.prompt
            | self.llm
            | self.output_parser
        )

    @staticmethod
    def stream_runnable(runnable: Runnable, inputs: Any, callbacks: Callbacks):
        final_output: Any = None
        for chunk in runnable.stream(inputs, config={"callbacks": callbacks}):
            if final_output is None:
                final_output = chunk
            else:
                final_output += chunk

        return final_output

    @staticmethod
    async def astream_runnable(runnable: Runnable, inputs: Any, callbacks: Callbacks):
        final_output: Any = None
        async for chunk in runnable.astream(inputs, config={"callbacks": callbacks}):
            if final_output is None:
                final_output = chunk
            else:
                final_output += chunk
        return final_output

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        inputs = {
            **kwargs,
            **{"agent_scratchpad": self.format_intermediate_steps(intermediate_steps)},
        }
        prompt_output = self.stream_runnable(
            runnable=self.prompt, inputs=inputs, callbacks=callbacks
        )
        llm_output = self.stream_runnable(
            runnable=self.llm, inputs=prompt_output, callbacks=callbacks
        )
        parsed_llm_output = self.stream_runnable(
            runnable=self.output_parser, inputs=llm_output, callbacks=callbacks
        )
        return parsed_llm_output

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        inputs = {
            **kwargs,
            **{"agent_scratchpad": self.format_intermediate_steps(intermediate_steps)},
        }
        prompt_output = await self.astream_runnable(
            runnable=self.prompt, inputs=inputs, callbacks=callbacks
        )
        llm_output = await self.astream_runnable(
            runnable=self.llm, inputs=prompt_output, callbacks=callbacks
        )
        parsed_llm_output = await self.astream_runnable(
            runnable=self.output_parser, inputs=llm_output, callbacks=callbacks
        )
        return parsed_llm_output


class PlanningLibOpenAIFunctionsAgent(PlanningLibBaseSingleActionAgent):
    output_parser = OpenAIFunctionsAgentOutputParser()

    @staticmethod
    def format_intermediate_steps(
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> List[BaseMessage]:
        return format_to_openai_function_messages(intermediate_steps)

    @staticmethod
    def convert_tool(tool: BaseTool) -> Any:
        return convert_to_openai_function(tool)
