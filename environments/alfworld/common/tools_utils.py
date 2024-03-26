from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from textworld.gym.envs.textworld_batch import TextworldBatchGymEnv  # type: ignore[import-untyped]


class EmptyInput(BaseModel): ...


class ObjectInput(BaseModel):
    object_type: str = Field(
        description="A type of the portable object.", examples=["the apple", "the mug"]
    )
    object_id: int = Field(
        description="A specific number associated with the object (e.g., when there are "
        "several mugs in the room, those would be mug 1 and mug 2).",
        examples=[1, 2, 3],
    )


class ReceptableInput(BaseModel):
    receptable_type: str = Field(
        description="A type of the receptable.",
        examples=["the coffee table", "the drawer", "the countertop"],
    )
    receptable_id: int = Field(
        description="A specific number associated with the receptable (e.g., when there are "
        "several drawers in the room, those would be drawer 1 and drawer 2).",
        examples=[1, 2, 3],
    )


class ObjectAndReceptableInput(BaseModel):
    object_type: str = Field(
        description="A type of the portable object.", examples=["the apple", "the mug"]
    )
    object_id: int = Field(
        description="A specific number associated with the object (e.g., when there are "
        "several mugs in the room, those would be mug 1 and mug 2).",
        examples=[1, 2, 3],
    )
    receptable_type: str = Field(
        description="A type of the receptable.",
        examples=["the coffee table", "the drawer", "the countertop"],
    )
    receptable_id: int = Field(
        description="A specific number associated with the receptable (e.g., when there are "
        "several drawers in the room, those would be drawer 1 and drawer 2).",
        examples=[1, 2, 3],
    )


class ObjectOrReceptableInput(BaseModel):
    type: str = Field(
        description="A type of the object (might be either a portable object or a static one).",
        examples=["the apple", "the coffee table"],
    )
    id: int = Field(
        description="A specific number associated with the object (e.g., when there are "
        "several mugs in the room, those would be mug 1 and mug 2).",
        examples=[1, 2, 3],
    )


class BaseALFWorldTool(BaseModel):
    """Base tool for an ALFWorld environment.

    Environment is present as a field, but it won't be shown to models."""

    env: TextworldBatchGymEnv = Field(exclude=True)

    class Config(BaseTool.Config):
        pass
