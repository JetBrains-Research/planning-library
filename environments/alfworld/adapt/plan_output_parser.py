import re

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from planning_library.strategies.adapt.utils import InitialADaPTPlan


class ALFWorldADaPTPPlanOutputParser(BaseOutputParser[InitialADaPTPlan]):
    def parse(self, text: str) -> InitialADaPTPlan:
        try:
            matches = re.findall(r"Step \d+: (.*)\n", text)
            return {
                "subtasks": [
                    {"inputs": {"inputs": match.rstrip()}} for match in matches
                ],
                "logic": "and",
            }
        except Exception as e:
            raise OutputParserException(f"Couldn't parse {text} due to {e}")
