from typing import Any, Dict, List, SupportsFloat, Tuple, Type

from langchain.pydantic_v1 import BaseModel
from langchain.tools import BaseTool
from textworld.gym.envs.textworld_batch import TextworldBatchGymEnv  # type: ignore[import-untyped]

from .tools_utils import (
    BaseALFWorldTool,
    EmptyInput,
    ObjectAndReceptableInput,
    ObjectOrReceptableInput,
    ReceptableInput,
)


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
    description = """Go to the specified receptable (static object)."""
    args_schema: Type[BaseModel] = ReceptableInput  # type: ignore

    def _run(
        self,
        receptable_type: str,
        receptable_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step([f"go to {receptable_type} {receptable_id}"])
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class OpenTool(BaseALFWorldTool, BaseTool):
    name = "open"
    description = """Open a specified receptable (static object). Only works when you're near a receptable and when it is closed."""
    args_schema: Type[BaseModel] = ReceptableInput  # type: ignore

    def _run(
        self,
        receptable_type: str,
        receptable_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step([f"open {receptable_type} {receptable_id}"])
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class CloseTool(BaseALFWorldTool, BaseTool):
    name = "close"
    description = """Close a specified receptable (static object). Only available when you're near a receptable and when it is closed."""
    args_schema: Type[BaseModel] = ReceptableInput  # type: ignore

    def _run(
        self,
        receptable_type: str,
        receptable_id: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step([f"close {receptable_type} {receptable_id}"])
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}


class TakeTool(BaseALFWorldTool, BaseTool):
    name = "take"
    description = """Pick up the specified portable object from the specified receptable (static object). Only works when you're near the specified receptable and the specified object is present in/on the receptable."""
    args_schema: Type[BaseModel] = ObjectAndReceptableInput  # type: ignore

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
    description = """Put the specified portable object in/on the specified receptable (static object). Only available when you're near the specified receptable and carry the specified portable object in your inventory."""
    args_schema: Type[BaseModel] = ObjectAndReceptableInput  # type: ignore

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
    description = """Toggle the specified object on/off (can be either a portable object or a static receptable). Only available when you're near the specified receptable/portable object or carry the specified portable object."""
    args_schema: Type[BaseModel] = ObjectOrReceptableInput  # type: ignore

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
    description = """Heat the portable object via the receptable (static object). Only available when you're already near the receptable and the portable object is in/on the receptable."""
    args_schema: Type[BaseModel] = ObjectAndReceptableInput  # type: ignore

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
    description = """Cool the portable object via the receptable (static object). Only available when you're already near a receptable and the portable object is in/on the receptable."""
    args_schema: Type[BaseModel] = ObjectAndReceptableInput  # type: ignore

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
    description = """Clean the portable object via the receptable (static object). Only available when you're already near a receptable and the portable object is in/on the receptable."""
    args_schema: Type[BaseModel] = ObjectAndReceptableInput  # type: ignore

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
    description = """Examine the specified object (can be either a portable object or a static receptable). Only available when you're near the receptable/portable object or carry the specified portable object."""
    args_schema: Type[BaseModel] = ObjectOrReceptableInput  # type: ignore

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
    args_schema: Type[BaseModel] = EmptyInput  # type: ignore

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
    args_schema: Type[BaseModel] = EmptyInput  # type: ignore

    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self.env.step(["look"])
        return obs[0], scores[0], dones[0], False, {key: infos[key][0] for key in infos}
