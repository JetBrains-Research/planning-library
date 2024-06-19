from dataclasses import dataclass, fields
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool


@dataclass
class MetaTools:
    reset: Optional[BaseTool] = None

    @property
    def tools(self) -> List[BaseTool]:
        tools: List[BaseTool] = []

        for field in fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, BaseTool):
                tools.append(field_value)

        return tools

    @property
    def tool_names_map(self) -> Dict[str, str]:
        tool_map: Dict[str, str] = {}

        for field in fields(self):
            field_value = getattr(self, field.name)
            if isinstance(field_value, BaseTool):
                tool_map[field.name] = field_value.name

        return tool_map
