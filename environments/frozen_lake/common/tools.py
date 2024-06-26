from textwrap import dedent
from typing import Any, Dict, Literal, SupportsFloat, Tuple, Type

import gymnasium as gym
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool


class BaseFrozenLakeTool(BaseModel):
    """Base tool for a FrozenLake environment.

    Environment is present as a field, but it won't be shown to models."""

    env: gym.Env = Field(exclude=True)

    class Config(BaseTool.Config):
        pass


class MoveInput(BaseModel):
    direction: Literal["left", "right", "down", "up"] = Field(description="Which direction to move.")


class MoveTool(BaseFrozenLakeTool, BaseTool):
    name = "move"
    description = dedent(
        """
    Moves one step in given direction. Returns the following:
    * observation: current position on the board;
    * reward: 1 when the goal is reached, 0 otherwise;
    * terminated: if True, the game has ended: there's no opportunity to move anymore (either the goal was found or the player has fallen into a hole);
    * truncated: if True, the time limit has been exceeded;
    * info: probability of moving in the wrong direction for the current cell (ice is slippery!)"""
    )
    args_schema: Type[BaseModel] = MoveInput  # type: ignore

    @staticmethod
    def _convert_frozenlake_observation_to_position(observation: int, nrow: int) -> Tuple[int, int]:
        # FrozenLake: observation = current_row * nrow + current_col
        current_row, current_col = observation // nrow, observation % nrow
        return current_col, current_row

    @staticmethod
    def _convert_direction_to_frozenlake(direction: str) -> int:
        if direction == "left":
            return 0
        elif direction == "down":
            return 1
        elif direction == "right":
            return 2
        elif direction == "up":
            return 3
        else:
            raise ValueError(f"Wrong tool input {direction}.")

    def _run(
        self,
        direction: str,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tuple[int, int], SupportsFloat, bool, bool, Dict[str, Any]]:
        _observation, reward, terminated, truncated, info = self.env.unwrapped.step(
            MoveTool._convert_direction_to_frozenlake(direction)
        )
        nrow = self.env.get_wrapper_attr("nrow")
        observation = MoveTool._convert_frozenlake_observation_to_position(observation=_observation, nrow=nrow)
        return observation, reward, terminated, truncated, info


class LookInput(BaseModel):
    direction: Literal["left", "right", "down", "up"] = Field(description="Which direction to look at.")


class LookTool(BaseFrozenLakeTool, BaseTool):
    name = "look"
    description = dedent("""
    Peeks at the adjacent cell in given direction. The following options are possible:
    * out of bounds - it's not possible to move in the given direction from the current cell;
    * S - starting cell;
    * H - hole;
    * F - frozen cell;
    * G - goal.
    """)
    args_schema: Type[BaseModel] = LookInput  # type: ignore

    def _run(
        self,
        direction: str,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        nrow = self.env.get_wrapper_attr("nrow")
        board = self.env.get_wrapper_attr("desc")
        x, y = MoveTool._convert_frozenlake_observation_to_position(
            observation=self.env.get_wrapper_attr("s"), nrow=nrow
        )

        if direction == "left":
            observation = "out of bounds" if x == 0 else board[x - 1][y].decode()
        elif direction == "right":
            observation = "out of bounds" if x == nrow - 1 else board[x + 1][y].decode()
        elif direction == "down":
            observation = "out of bounds" if y == nrow - 1 else board[x][y + 1].decode()
        elif direction == "up":
            observation = "out of bounds" if y == 0 else board[x][y - 1].decode()
        else:
            raise ValueError("Wrong direction; expected one of: 'left', 'right', 'down', 'up'.")

        info: Dict[str, Any]
        reward, terminated, truncated, info = (
            0,
            False,
            False,
            {},
        )

        return observation, reward, terminated, truncated, info


class CheckMapInput(BaseModel): ...


class CheckMapTool(BaseFrozenLakeTool, BaseTool):
    name = "check_map"
    description = dedent("""
    Peeks at current map without changing its state. 

    The map is an n x n grid where different types of cells are denoted by different letters:
    * S - start cell
    * G - goal cell
    * F - frozen cell
    * H - hole cell
    
    Example for 2 x 2 case:
    
    SH
    FG
    """)
    args_schema: Type[BaseModel] = CheckMapInput  # type: ignore

    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[str, SupportsFloat, bool, bool, Dict[str, Any]]:
        info: Dict[str, Any]
        observation, reward, terminated, truncated, info = (
            "\n".join("".join(x.decode() for x in y) for y in self.env.get_wrapper_attr("desc")),
            0,
            False,
            False,
            {},
        )
        return observation, reward, terminated, truncated, info


class CheckPositionInput(BaseModel): ...


class CheckPositionTool(BaseFrozenLakeTool, BaseTool):
    name = "check_position"
    description = """Peeks at current position map without changing its state."""
    args_schema: Type[BaseModel] = CheckMapInput  # type: ignore

    def _run(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[Tuple[int, int], SupportsFloat, bool, bool, Dict[str, Any]]:
        info: Dict[str, Any]
        observation, reward, terminated, truncated, info = (
            MoveTool._convert_frozenlake_observation_to_position(
                self.env.get_wrapper_attr("s"), nrow=self.env.get_wrapper_attr("nrow")
            ),
            0,
            False,
            False,
            {},
        )
        return observation, reward, terminated, truncated, info
