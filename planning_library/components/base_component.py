from typing import Generic, TypeVar, Optional, Mapping, Set, Dict, Callable, Awaitable
from abc import ABC, abstractmethod
from langchain_core.callbacks import CallbackManager, AsyncCallbackManager
from langchain_core.prompts import ChatPromptTemplate

InputType = TypeVar("InputType", bound=Mapping)
OutputType = TypeVar("OutputType")


class BaseComponent(Generic[InputType, OutputType], ABC):
    name: Optional[str] = None
    required_prompt_input_vars: Set[str] = set()

    @classmethod
    def _create_default_prompt(
        cls, system_message: Optional[str], user_message: str
    ) -> ChatPromptTemplate:
        raise NotImplementedError(
            f"Default prompt is not supported for {cls.__name__}. Please, provide `prompt` instead of `user_message`."
        )

    @classmethod
    def _process_prompt(
        cls,
        prompt: Optional[ChatPromptTemplate] = None,
        system_message: Optional[str] = None,
        user_message: Optional[str] = None,
    ) -> ChatPromptTemplate:
        if prompt is None:
            if user_message is None:
                raise ValueError(
                    "Either `prompt` or `user_message` are required to create an agent."
                )
            prompt = cls._create_default_prompt(
                system_message=system_message, user_message=user_message
            )

        missing_vars = cls.required_prompt_input_vars.difference(prompt.input_variables)
        if missing_vars:
            raise ValueError(
                f"Prompt for {cls.__name__} missing required variables: {missing_vars}"
            )

        return prompt

    def add_input_preprocessing(
        self,
        preprocess: Callable[[InputType], Dict],
        apreprocess: Optional[Callable[[InputType], Awaitable[Dict]]] = None,
    ) -> None: ...

    def add_output_preprocessing(
        self,
        preprocess: Callable[[OutputType], OutputType],
        apreprocess: Optional[Callable[[OutputType], Awaitable[OutputType]]] = None,
    ) -> None: ...

    @abstractmethod
    def invoke(
        self, inputs: InputType, run_manager: Optional[CallbackManager] = None, **kwargs
    ) -> OutputType: ...

    @abstractmethod
    async def ainvoke(
        self,
        inputs: InputType,
        run_manager: Optional[AsyncCallbackManager] = None,
        **kwargs,
    ) -> OutputType: ...
