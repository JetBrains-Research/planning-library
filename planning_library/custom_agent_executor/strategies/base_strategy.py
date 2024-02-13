from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import AgentExecutor
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping


class BaseStrategy(AgentExecutor, ABC):
    @abstractmethod
    def _run_strategy(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[AgentFinish, List[Tuple[AgentAction, str]]]:
        ...

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""

        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping([tool.name for tool in self.tools], excluded_colors=["green", "red"])

        output, intermediate_steps = self._run_strategy(
            name_to_tool_map,
            color_mapping,
            inputs,
            run_manager=run_manager,
        )

        return self._return(output, intermediate_steps, run_manager=run_manager)
