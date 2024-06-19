from typing import Dict, List, Type, Union

from planning_library.function_calling_parsers import (
    BaseFunctionCallingMultiActionParser,
    BaseFunctionCallingSingleActionParser,
)


class ParserRegistry:
    registry: Dict[
        str,
        Union[BaseFunctionCallingSingleActionParser, BaseFunctionCallingMultiActionParser],
    ] = {}

    @classmethod
    def get_parser(
        cls, parser_name
    ) -> Union[BaseFunctionCallingSingleActionParser, BaseFunctionCallingMultiActionParser]:
        try:
            return cls.registry[parser_name]
        except KeyError:
            raise ValueError(f"Unknown parser {parser_name}. Currently available are: {cls.get_available_parsers()}")

    @classmethod
    def get_available_parsers(cls) -> List[str]:
        return list(cls.registry.keys())

    @classmethod
    def register(
        cls,
        other_cls: Union[
            Type[BaseFunctionCallingSingleActionParser],
            Type[BaseFunctionCallingMultiActionParser],
        ],
    ):
        cls.registry[other_cls.name] = other_cls()
        return other_cls
