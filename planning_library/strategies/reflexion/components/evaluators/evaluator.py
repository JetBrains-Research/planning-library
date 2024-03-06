from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager

from .backbones import ReflexionBaseEvaluatorBackbone
from .continue_judges import ReflexionBaseEvaluatorContinueJudge


class ReflexionEvaluator:
    def __init__(self, backbone: ReflexionBaseEvaluatorBackbone, judge: ReflexionBaseEvaluatorContinueJudge):
        self.backbone = backbone
        self.judge = judge

    def evaluate(
        self,
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        agent_outcome: Union[List[AgentAction], AgentAction, AgentFinish],
        run_manager: Optional[CallbackManager] = None,
    ) -> Tuple[Any, bool]:
        value = self.backbone.evaluate(
            inputs=inputs,
            intermediate_steps=intermediate_steps,
            agent_outcome=agent_outcome,
            run_manager=run_manager,
        )
        should_continue = self.judge.should_continue(
            inputs=inputs,
            intermediate_steps=intermediate_steps,
            agent_outcome=agent_outcome,
            run_manager=run_manager,
            value=value,
        )
        return value, should_continue

    async def aevaluate(
        self,
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        agent_outcome: Union[List[AgentAction], AgentAction, AgentFinish],
        run_manager: Optional[CallbackManager] = None,
    ) -> Tuple[Any, bool]:
        value = await self.backbone.aevaluate(
            inputs=inputs,
            intermediate_steps=intermediate_steps,
            agent_outcome=agent_outcome,
            run_manager=run_manager,
        )
        should_continue = await self.judge.ashould_continue(
            inputs=inputs,
            intermediate_steps=intermediate_steps,
            agent_outcome=agent_outcome,
            run_manager=run_manager,
            value=value,
        )
        return value, should_continue
