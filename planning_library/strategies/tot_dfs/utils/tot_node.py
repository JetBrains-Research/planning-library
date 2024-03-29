from typing import List, Optional, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish, AgentStep


class ToTNode:
    def __init__(
        self,
        parent: Optional["ToTNode"] = None,
        thought: Optional[Union[List[AgentAction], AgentAction, AgentFinish]] = None,
        observation: Optional[Union[List[AgentStep], AgentStep]] = None,
    ):
        self.parent = parent
        self.children: List["ToTNode"] = []
        self.thought = thought
        self.observation = observation

    @property
    def trajectory(self) -> List[Tuple[AgentAction, str]]:
        """Returns the (action, observation) tuples on the path from the root to the current node.

        Note:
            * The nodes are arranged in order from the root to the current node.
        """
        node: Optional[ToTNode] = self
        trajectory_actions: List[Tuple[AgentAction, str]] = []
        while node is not None:
            if isinstance(node.thought, list):
                assert isinstance(node.observation, list) and len(node.thought) == len(
                    node.observation
                )
                trajectory_actions.extend(
                    (observation.action, observation.observation)
                    for observation in node.observation
                )
            elif isinstance(node.thought, AgentAction):
                assert isinstance(node.observation, AgentStep)
                trajectory_actions.append(
                    (node.observation.action, node.observation.observation)
                )
            elif isinstance(node.thought, AgentFinish) and node is not self:
                raise ValueError("AgentFinish detected as non-terminal node.")

            node = node.parent
        return trajectory_actions[::-1]
