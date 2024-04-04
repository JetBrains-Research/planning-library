from typing import Generic, TypeVar, Optional, Dict
from abc import ABC, abstractmethod
from langchain_core.callbacks import CallbackManager, AsyncCallbackManager

InputType = TypeVar("InputType", bound=Dict)
OutputType = TypeVar("OutputType")


class BaseComponent(Generic[InputType, OutputType], ABC):
    @abstractmethod
    def invoke(
        self,
        inputs: InputType,
        run_manager: Optional[CallbackManager] = None,
    ) -> OutputType: ...

    @abstractmethod
    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
    ) -> OutputType: ...
