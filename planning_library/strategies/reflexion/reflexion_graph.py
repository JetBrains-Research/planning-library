from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, Union

from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph  # type: ignore[import]
from langgraph.pregel import Pregel  # type: ignore[import-untyped]

from ...action_executors import BaseActionExecutor
from ...utils import get_tools_maps
from .components.actors import BaseActor
from .components.evaluators import ReflexionEvaluator
from .components.self_reflections import BaseSelfReflection


class ReflexionState(TypedDict):
    """A state passed between nodes of Reflexion graph."""

    inputs: Dict[str, Any]
    agent_outcome: Optional[Union[List[AgentAction], AgentAction, AgentFinish]]
    evaluator_score: Any
    evaluator_should_continue: Optional[bool]
    self_reflections: List[str]
    intermediate_steps: List[Tuple[AgentAction, str]]
    iteration: int


class ReflexionNodes:
    @staticmethod
    def init(state: ReflexionState) -> ReflexionState:
        """The entry node in the graph. Initializes the state correctly."""
        state["agent_outcome"] = None
        state["evaluator_score"] = None
        state["evaluator_should_continue"] = None
        state["self_reflections"] = []
        state["intermediate_steps"] = []
        state["iteration"] = 1
        return state

    @staticmethod
    def re_init(state: ReflexionState, reset_environment: Optional[Callable[[Dict[str, Any]], None]]) -> ReflexionState:
        """The first node that gets called after at least one iteration. Handles the advance of the loop interation correctly."""
        state["agent_outcome"] = None
        state["evaluator_score"] = None
        state["evaluator_should_continue"] = None
        state["intermediate_steps"] = []
        state["iteration"] += 1

        if reset_environment:
            reset_environment(state["inputs"])

        return state

    @staticmethod
    def act(state: ReflexionState, actor: BaseActor) -> ReflexionState:
        """Synchronous version of calling an agent and returning its result."""
        agent_outcome = actor.act(
            inputs=state["inputs"],
            intermediate_steps=state["intermediate_steps"],
            self_reflections=state["self_reflections"],
        )
        state["agent_outcome"] = agent_outcome
        return state

    @staticmethod
    async def aact(state: ReflexionState, actor: BaseActor) -> ReflexionState:
        """Asynchronous version of calling an agent and returning its result."""
        agent_outcome = await actor.aact(
            inputs=state["inputs"],
            intermediate_steps=state["intermediate_steps"],
            self_reflections=state["self_reflections"],
        )

        state["agent_outcome"] = agent_outcome
        return state

    @staticmethod
    def execute_actions(
        state: ReflexionState,
        action_executor: BaseActionExecutor,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
    ) -> ReflexionState:
        """Synchronous version of executing actions as previously requested by an agent."""
        assert state["agent_outcome"] is not None, "Agent outcome should be defined on the tool execution step."
        assert not isinstance(
            state["agent_outcome"], AgentFinish
        ), "Agent outcome should not be AgentFinish on the tool execution step."

        observation = action_executor.execute(
            actions=state["agent_outcome"], name_to_tool_map=name_to_tool_map, color_mapping=color_mapping
        )

        if isinstance(observation, AgentStep):
            state["intermediate_steps"].append((observation.action, observation.observation))
        else:
            state["intermediate_steps"].extend([(obs.action, obs.observation) for obs in observation])
        return state

    @staticmethod
    async def aexecute_actions(
        state: ReflexionState,
        action_executor: BaseActionExecutor,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
    ) -> ReflexionState:
        """Asynchronous version of executing tools as previously requested by an agent."""
        assert state["agent_outcome"] is not None, "Agent outcome should be defined on the tool execution step."
        assert not isinstance(
            state["agent_outcome"], AgentFinish
        ), "Agent outcome should not be AgentFinish on the tool execution step."

        observation = await action_executor.aexecute(
            actions=state["agent_outcome"], name_to_tool_map=name_to_tool_map, color_mapping=color_mapping
        )

        if isinstance(observation, AgentStep):
            state["intermediate_steps"].append((observation.action, observation.observation))
        else:
            state["intermediate_steps"].extend([(obs.action, obs.observation) for obs in observation])
        return state

    @staticmethod
    def evaluate(state: ReflexionState, evaluator: ReflexionEvaluator) -> ReflexionState:
        """Synchronous version of evaluating the outcome of the current trial."""
        assert isinstance(
            state["agent_outcome"], AgentFinish
        ), "Agent outcome should be AgentFinish on the evaluation step."
        value, should_continue = evaluator.evaluate(
            inputs=state["inputs"], intermediate_steps=state["intermediate_steps"], agent_outcome=state["agent_outcome"]
        )
        state["evaluator_score"] = value
        state["evaluator_should_continue"] = should_continue
        return state

    @staticmethod
    async def aevaluate(state: ReflexionState, evaluator: ReflexionEvaluator) -> ReflexionState:
        """Asynchronous version of evaluating the outcome of the current trial."""
        assert isinstance(
            state["agent_outcome"], AgentFinish
        ), "Agent outcome should be AgentFinish on the evaluation step."

        value, should_continue = await evaluator.aevaluate(
            inputs=state["inputs"], intermediate_steps=state["intermediate_steps"], agent_outcome=state["agent_outcome"]
        )
        state["evaluator_score"] = value
        state["evaluator_should_continue"] = should_continue
        return state

    @staticmethod
    def self_reflect(state: ReflexionState, self_reflection: BaseSelfReflection) -> ReflexionState:
        """Synchronous version of self-reflecting on the current trial."""
        assert isinstance(
            state["agent_outcome"], AgentFinish
        ), "Agent outcome should be AgentFinish on the self-reflection step."

        reflection = self_reflection.self_reflect(
            inputs=state["inputs"],
            intermediate_steps=state["intermediate_steps"],
            agent_outcome=state["agent_outcome"],
            evaluator_score=state["evaluator_score"],
        )
        state["self_reflections"].append(reflection)
        return state

    @staticmethod
    async def aself_reflect(state: ReflexionState, self_reflection: BaseSelfReflection) -> ReflexionState:
        """Asynchronous version of self-reflecting on the current trial."""
        assert isinstance(
            state["agent_outcome"], AgentFinish
        ), "Agent outcome should be AgentFinish on the self-reflection step."

        reflection = await self_reflection.aself_reflect(
            inputs=state["inputs"],
            intermediate_steps=state["intermediate_steps"],
            agent_outcome=state["agent_outcome"],
            evaluator_score=state["evaluator_score"],
        )
        state["self_reflections"].append(reflection)
        return state


class ReflexionEdges:
    @staticmethod
    def should_continue_evaluator(state: ReflexionState) -> Literal["yes", "no"]:
        """Conditional edge that determines whether the main loop should be continued or stopped based on evaluator's output.

        If yes: execution finishes.
        If no: next iteration.
        """
        return "yes" if state["evaluator_should_continue"] else "no"

    @staticmethod
    def should_continue_actor(state: ReflexionState) -> Literal["yes", "no"]:
        """Conditional edge that determines whether the agent loop should be continued or stopped based on agent's output.

        If yes: proceed to evaluation.
        If no: proceed to act.
        """
        return "no" if isinstance(state["agent_outcome"], AgentFinish) else "yes"

    @staticmethod
    def should_continue_num_iterations(
        state: ReflexionState,
        max_iterations: Optional[int],
    ) -> Literal["yes", "no"]:
        """Conditional edge that determines whether the main loop should be continued or stopped based on the number of iterations threshold.

        If yes: execution finishes.
        If no: next iteration.
        """
        if max_iterations is not None and state["iteration"] >= max_iterations:
            return "no"
        return "yes"


def create_reflexion_graph(
    actor: BaseActor,
    evaluator: ReflexionEvaluator,
    self_reflection: BaseSelfReflection,
    action_executor: BaseActionExecutor,
    tools: Sequence[BaseTool],
    max_iterations: Optional[int],
    reset_environment: Optional[Callable[[Dict[str, Any]], None]],
) -> Pregel:
    """Builds a graph for Reflexion strategy."""

    builder = StateGraph(ReflexionState)
    builder.add_node("init", ReflexionNodes.init)
    builder.add_node(
        "re_init",
        partial(ReflexionNodes.re_init, reset_environment=reset_environment),
    )
    builder.add_node(
        "act", RunnableLambda(partial(ReflexionNodes.act, actor=actor), afunc=partial(ReflexionNodes.aact, actor=actor))
    )

    name_to_tool_map, color_mapping = get_tools_maps(tools)
    builder.add_node(
        "execute_actions",
        RunnableLambda(
            partial(
                ReflexionNodes.execute_actions,
                action_executor=action_executor,
                name_to_tool_map=name_to_tool_map,
                color_mapping=color_mapping,
            ),
            afunc=partial(
                ReflexionNodes.aexecute_actions,
                action_executor=action_executor,
                name_to_tool_map=name_to_tool_map,
                color_mapping=color_mapping,
            ),
        ),
    )
    builder.add_node(
        "evaluate",
        RunnableLambda(
            partial(ReflexionNodes.evaluate, evaluator=evaluator),
            afunc=partial(ReflexionNodes.aevaluate, evaluator=evaluator),
        ),
    )
    builder.add_node(
        "self_reflect",
        RunnableLambda(
            partial(ReflexionNodes.self_reflect, self_reflection=self_reflection),
            afunc=partial(ReflexionNodes.aself_reflect, self_reflection=self_reflection),
        ),
    )

    builder.set_entry_point("init")
    builder.add_edge("init", "act")

    builder.add_conditional_edges(
        "act", ReflexionEdges.should_continue_actor, {"yes": "execute_actions", "no": "evaluate"}
    )
    builder.add_edge("execute_actions", "act")
    builder.add_conditional_edges(
        "evaluate",
        ReflexionEdges.should_continue_evaluator,
        {
            "yes": "self_reflect",
            "no": END,
        },
    )
    builder.add_conditional_edges(
        "self_reflect",
        partial(ReflexionEdges.should_continue_num_iterations, max_iterations=max_iterations),
        {"yes": "re_init", "no": END},
    )
    builder.add_edge("re_init", "act")
    return builder.compile()
