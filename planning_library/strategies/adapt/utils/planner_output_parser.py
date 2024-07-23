from langchain_core.output_parsers import BaseOutputParser
from langchain_core.utils.json import parse_and_check_json_markdown

from .typing_utils import ADaPTPlannerOutput


class SimplePlannerOutputParser(BaseOutputParser[ADaPTPlannerOutput]):
    def parse(self, text: str) -> ADaPTPlannerOutput:
        output = parse_and_check_json_markdown(text, expected_keys=["subtasks", "aggregation_mode"])
        return {
            "subtasks": output["subtasks"],
            "aggregation_mode": output["aggregation_mode"],
        }
