from langchain.pydantic_v1 import BaseModel
from langchain.tools import BaseTool
from typing import Type, Any, Tuple, Dict, List

from gymnasium.core import SupportsFloat
from .tools_utils import (
    BaseALFWorldTool,
    ReceptableInput,
    ObjectOrReceptableInput,
    ObjectAndReceptableInput,
    EmptyInput,
)
from textworld.gym.envs.textworld_batch import TextworldBatchGymEnv  # type: ignore[import-untyped]


def get_alfworld_tools(env: TextworldBatchGymEnv) -> List[BaseTool]:
    return [
        GoToTool(env=env),  # type: ignore[call-arg]
        OpenTool(env=env),  # type: ignore[call-arg]
        CloseTool(env=env),  # type: ignore[call-arg]
        TakeTool(env=env),  # type: ignore[call-arg]
        PutTool(env=env),  # type: ignore[call-arg]
        ToggleTool(env=env),  # type: ignore[call-arg]
        HeatTool(env=env),  # type: ignore[call-arg]
        CoolTool(env=env),  # type: ignore[call-arg]
        CleanTool(env=env),  # type: ignore[call-arg]
        ExamineTool(env=env),  # type: ignore[call-arg]
        InventoryTool(env=env),  # type: ignore[call-arg]
        LookTool(env=env),  # type: ignore[call-arg]
    ]


class GoToTool(BaseALFWorldTool, BaseTool):
    name = "goto"
    description = """Go to a specified receptable (static object)."""
    args_schema: Type[BaseModel] = ReceptableInput

    def _run(
        self,
        receptable_type: str,
        receptable_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step(
            [f"go to {receptable_type} {receptable_id}"]
        )
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class OpenTool(BaseALFWorldTool, BaseTool):
    name = "open"
    description = """Open a specified receptable (static object). Only available when you're already near a receptable."""
    args_schema: Type[BaseModel] = ReceptableInput

    def _run(
        self,
        receptable_type: str,
        receptable_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step(
            [f"open {receptable_type} {receptable_id}"]
        )
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class CloseTool(BaseALFWorldTool, BaseTool):
    name = "close"
    description = """Close a specified receptable (static object). Only available when you're already near a receptable."""
    args_schema: Type[BaseModel] = ReceptableInput

    def _run(
        self,
        receptable_type: str,
        receptable_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step(
            [f"close {receptable_type} {receptable_id}"]
        )
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class TakeTool(BaseALFWorldTool, BaseTool):
    name = "take"
    description = """Pick up a specified portable object from a specified receptable (static object). Only available when you're already near a receptable."""
    args_schema: Type[BaseModel] = ObjectAndReceptableInput

    def _run(
        self,
        object_type: str,
        object_id: int,
        receptable_type: str,
        receptable_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step(
            [f"take {object_type} {object_id} from {receptable_type} {receptable_id}"]
        )
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class PutTool(BaseALFWorldTool, BaseTool):
    name = "put"
    description = """Put a specified portable object in/щт a specified receptable (static object). Only available when you're already near a receptable and carry a portable object in your inventory."""
    args_schema: Type[BaseModel] = ObjectAndReceptableInput

    def _run(
        self,
        object_type: str,
        object_id: int,
        receptable_type: str,
        receptable_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step(
            [f"put {object_type} {object_id} in/on {receptable_type} {receptable_id}"]
        )
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class ToggleTool(BaseALFWorldTool, BaseTool):
    name = "toggle"
    description = """Toggle a specified object on/off (can be either a portable object or a static receptable). Only available when you're already near a receptable/a portable object or carry a portable object."""
    args_schema: Type[BaseModel] = ObjectOrReceptableInput

    def _run(
        self,
        type: str,
        id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step([f"toggle {type} {id}"])
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class HeatTool(BaseALFWorldTool, BaseTool):
    name = "heat"
    description = """Heat a portable object via a receptable (static object). Only available when you're already near a receptable and carry a portable object."""
    args_schema: Type[BaseModel] = ObjectAndReceptableInput

    def _run(
        self,
        object_type: str,
        object_id: int,
        receptable_type: str,
        receptable_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step(
            [f"heat {object_type} {object_id} with {receptable_type} {receptable_id}"]
        )
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class CoolTool(BaseALFWorldTool, BaseTool):
    name = "cool"
    description = """Cool a portable object via a receptable (static object). Only available when you're already near a receptable and carry a portable object."""
    args_schema: Type[BaseModel] = ObjectAndReceptableInput

    def _run(
        self,
        object_type: str,
        object_id: int,
        receptable_type: str,
        receptable_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step(
            [f"cool {object_type} {object_id} with {receptable_type} {receptable_id}"]
        )
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class CleanTool(BaseALFWorldTool, BaseTool):
    name = "clean"
    description = """Clean a portable object via a receptable (static object). Only available when you're already near a receptable and a portable object or carry a portable object."""
    args_schema: Type[BaseModel] = ObjectAndReceptableInput

    def _run(
        self,
        object_type: str,
        object_id: int,
        receptable_type: str,
        receptable_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step(
            [f"clean {object_type} {object_id} with {receptable_type} {receptable_id}"]
        )
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class ExamineTool(BaseALFWorldTool, BaseTool):
    name = "examine"
    description = """Examine a specified object (can be either a portable object or a static receptable). Only available when you're already near a receptable/a portable object or carry a portable object."""
    args_schema: Type[BaseModel] = ObjectOrReceptableInput

    def _run(
        self,
        type: str,
        id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step([f"examine {type} {id}"])
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class InventoryTool(BaseALFWorldTool, BaseTool):
    name = "inventory"
    description = """Check if you are carrying any portable objects."""
    args_schema: Type[BaseModel] = EmptyInput

    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step(["inventory"])
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class LookTool(BaseALFWorldTool, BaseTool):
    name = "look"
    description = """Check your surroundings."""
    args_schema: Type[BaseModel] = EmptyInput

    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step(["look"])
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}
