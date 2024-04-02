from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import AsyncCallbackManager, CallbackManager


class BaseThoughtSorter(ABC):
    @abstractmethod
    def sort_thoughts(
        self,
        thoughts: List[List[AgentAction] | AgentFinish]
        | List[AgentAction | AgentFinish],
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManager] = None,
    ) -> List[List[AgentAction] | AgentFinish] | List[AgentAction | AgentFinish]: ...

    @abstractmethod
    async def asort_thoughts(
        self,
        thoughts: List[List[AgentAction] | AgentFinish]
        | List[AgentAction | AgentFinish],
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> List[List[AgentAction] | AgentFinish] | List[AgentAction | AgentFinish]: ...
