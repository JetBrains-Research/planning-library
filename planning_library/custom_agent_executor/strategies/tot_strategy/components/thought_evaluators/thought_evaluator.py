from typing import Dict, List, Optional, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import CallbackManager

from .backbones import BaseThoughtEvaluatorBackbone
from .continue_judges import BaseThoughtEvaluatorContinueJudge


class ThoughtEvaluator:
    """A thought evaluator.

    It is responsible for two things:
    * evaluating each proposed thought;
    * determining based on the value if the thought should be explored further or discarded.
    """

    def __init__(self, backbone: BaseThoughtEvaluatorBackbone, judge: BaseThoughtEvaluatorContinueJudge):
        self.backbone = backbone
        self.judge = judge

    def evaluate(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        next_thought: Union[List[AgentAction], AgentAction, AgentFinish],
        observation: Optional[Union[List[AgentStep], AgentStep]],
        run_manager: Optional[CallbackManager] = None,
    ) -> bool:
        value = self.backbone.evaluate(
            inputs=inputs,
            trajectory=trajectory,
            next_thought=next_thought,
            observation=observation,
            run_manager=run_manager,
        )
        should_continue = self.judge.should_continue(
            inputs=inputs, trajectory=trajectory, next_thought=next_thought, run_manager=run_manager, value=value
        )
        return should_continue

    async def aevaluate(
        self,
        inputs: Dict[str, str],
        trajectory: List[Tuple[AgentAction, str]],
        next_thought: Union[AgentAction, AgentFinish],
        observation: Optional[Union[List[AgentStep], AgentStep]],
        run_manager: Optional[CallbackManager] = None,
    ) -> bool:
        value = await self.backbone.aevaluate(
            inputs=inputs,
            trajectory=trajectory,
            next_thought=next_thought,
            observation=observation,
            run_manager=run_manager,
        )
        should_continue = await self.judge.ashould_continue(
            inputs=inputs, trajectory=trajectory, next_thought=next_thought, run_manager=run_manager, value=value
        )
        return should_continue