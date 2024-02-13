from typing import List, Union

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.output_parsers import BaseOutputParser

game_of_24_generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that proposes next steps in Game of 24."),
        (
            "human",
            "Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.",
        ),
        FewShotChatMessagePromptTemplate(
            example_prompt=ChatPromptTemplate.from_messages(
                [
                    ("human", "Input: {inputs}\n{max_num_thoughts} variants for a possible next step:"),
                    ("ai", "{thought}"),
                ]
            ),
            examples=[
                {
                    "inputs": "2 8 8 14",
                    "max_num_thoughts": "8",
                    "thought": "2 + 8 = 10 (left: 8 10 14)\n8 / 2 = 4 (left: 4 8 14)\n14 + 2 = 16 (left: 8 8 16)\n2 * 8 = 16 (left: 8 14 16)\n8 - 2 = 6 (left: 6 8 14)\n14 - 8 = 6 (left: 2 6 8)\n14 /  2 = 7 (left: 7 8 8)\n14 - 2 = 12 (left: 8 8 12)",
                },
                {
                    "inputs": "10 14",
                    "max_num_thoughts": "1",
                    "thought": "10 + 14 = 24",
                },
            ],
        ),
        ("human", "Input: {inputs}\n{max_num_thoughts} variants for a possible next step:"),
    ]
)


class GameOf24GenerateOutputParser(BaseOutputParser[List[Union[AgentAction, AgentFinish]]]):
    def parse(self, text: str) -> List[Union[AgentAction, AgentFinish]]:
        outputs: List[Union[AgentAction, AgentFinish]] = []
        for line in text.split("\n"):
            if line.endswith("= 24"):
                outputs.append(AgentFinish({"output": line}, line))
            else:
                outputs.append(AgentAction(tool="simple_tool", tool_input=line, log=line))

        return outputs
