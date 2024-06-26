from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

from langchain.memory import ChatMessageHistory
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph  # type: ignore[import]
from langgraph.pregel import Pregel  # type: ignore[import-untyped]

from ...action_executors import BaseActionExecutor
from .components import (
    ReflexionActor,
    ReflexionActorInput,
    ReflexionEvaluator,
    ReflexionEvaluatorInput,
    ReflexionSelfReflection,
    ReflexionSelfReflectionInput,
)


class ReflexionState(TypedDict):
    """A state passed between nodes of Reflexion graph."""

    inputs: Dict[str, Any]
    agent_outcome: Optional[Union[List[AgentAction], AgentAction, AgentFinish]]
    evaluator_should_continue: Optional[bool]
    self_reflection_memory: BaseChatMessageHistory
    self_reflections: Sequence[BaseMessage]
    intermediate_steps: List[Tuple[AgentAction, str]]
    iteration: int


class ReflexionNodes:
    @staticmethod
    def _format_self_reflections(
        self_reflections: List[Tuple[str, str]],
    ) -> Sequence[BaseMessage]:
        result = []
        for t in self_reflections:
            if t[0] == "content":
                content = t[1]
                message = AIMessage(content=content)
                result.append(message)

        return result

    @staticmethod
    def init(state: ReflexionState, memory: Optional[BaseChatMessageHistory] = None) -> ReflexionState:
        """The entry node in the graph. Initializes the state correctly."""
        state["agent_outcome"] = None
        state["evaluator_should_continue"] = None
        state["self_reflection_memory"] = ChatMessageHistory() if memory is None else memory
        state["self_reflections"] = []
        state["intermediate_steps"] = []
        state["iteration"] = 1
        return state

    @staticmethod
    def re_init(
        state: ReflexionState,
        reset_environment: Optional[Callable[[Dict[str, Any]], None]],
    ) -> ReflexionState:
        """The first node that gets called after at least one iteration. Handles the advance of the loop interation correctly."""
        state["agent_outcome"] = None
        state["evaluator_should_continue"] = None
        state["intermediate_steps"] = []
        state["iteration"] += 1
        # TODO: why does memory return list of tuples instead of messages as expected? some serialization stuff?
        state["self_reflections"] = ReflexionNodes._format_self_reflections(
            state["self_reflection_memory"].messages  # type: ignore[arg-type]
        )

        if reset_environment:
            reset_environment(state["inputs"])

        return state

    @staticmethod
    def act(state: ReflexionState, actor: ReflexionActor) -> ReflexionState:
        """Synchronous version of calling an agent and returning its result."""
        agent_outcome = actor.invoke(
            ReflexionActorInput(
                inputs=state["inputs"],
                intermediate_steps=state["intermediate_steps"],
                self_reflections=state["self_reflections"],
            )
        )
        state["agent_outcome"] = agent_outcome
        return state

    @staticmethod
    async def aact(state: ReflexionState, actor: ReflexionActor) -> ReflexionState:
        """Asynchronous version of calling an agent and returning its result."""
        agent_outcome = await actor.ainvoke(
            ReflexionActorInput(
                inputs=state["inputs"],
                intermediate_steps=state["intermediate_steps"],
                self_reflections=state["self_reflections"],
            )
        )

        state["agent_outcome"] = agent_outcome
        return state

    @staticmethod
    def execute_actions(
        state: ReflexionState,
        action_executor: BaseActionExecutor,
    ) -> ReflexionState:
        """Synchronous version of executing actions as previously requested by an agent."""
        assert state["agent_outcome"] is not None, "Agent outcome should be defined on the tool execution step."
        assert not isinstance(
            state["agent_outcome"], AgentFinish
        ), "Agent outcome should not be AgentFinish on the tool execution step."

        observation = action_executor.execute(
            actions=state["agent_outcome"],
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
    ) -> ReflexionState:
        """Asynchronous version of executing tools as previously requested by an agent."""
        assert state["agent_outcome"] is not None, "Agent outcome should be defined on the tool execution step."
        assert not isinstance(
            state["agent_outcome"], AgentFinish
        ), "Agent outcome should not be AgentFinish on the tool execution step."

        observation = await action_executor.aexecute(
            actions=state["agent_outcome"],
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
        should_continue = evaluator.invoke(
            ReflexionEvaluatorInput(
                inputs=state["inputs"],
                intermediate_steps=state["intermediate_steps"],
                agent_outcome=state["agent_outcome"],
            )
        )
        state["evaluator_should_continue"] = should_continue
        return state

    @staticmethod
    async def aevaluate(state: ReflexionState, evaluator: ReflexionEvaluator) -> ReflexionState:
        """Asynchronous version of evaluating the outcome of the current trial."""
        assert isinstance(
            state["agent_outcome"], AgentFinish
        ), "Agent outcome should be AgentFinish on the evaluation step."

        should_continue = await evaluator.ainvoke(
            ReflexionEvaluatorInput(
                inputs=state["inputs"],
                intermediate_steps=state["intermediate_steps"],
                agent_outcome=state["agent_outcome"],
            )
        )
        state["evaluator_should_continue"] = should_continue
        return state

    @staticmethod
    def self_reflect(state: ReflexionState, self_reflection: ReflexionSelfReflection) -> ReflexionState:
        """Synchronous version of self-reflecting on the current trial."""
        assert isinstance(
            state["agent_outcome"], AgentFinish
        ), "Agent outcome should be AgentFinish on the self-reflection step."

        reflection = self_reflection.invoke(
            ReflexionSelfReflectionInput(
                inputs=state["inputs"],
                intermediate_steps=state["intermediate_steps"],
                agent_outcome=state["agent_outcome"],
            )
        )
        state["self_reflection_memory"].add_messages(reflection)
        return state

    @staticmethod
    async def aself_reflect(state: ReflexionState, self_reflection: ReflexionSelfReflection) -> ReflexionState:
        """Asynchronous version of self-reflecting on the current trial."""
        assert isinstance(
            state["agent_outcome"], AgentFinish
        ), "Agent outcome should be AgentFinish on the self-reflection step."

        reflection = await self_reflection.ainvoke(
            ReflexionSelfReflectionInput(
                inputs=state["inputs"],
                intermediate_steps=state["intermediate_steps"],
                agent_outcome=state["agent_outcome"],
            )
        )
        await state["self_reflection_memory"].aadd_messages(reflection)
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
    actor: ReflexionActor,
    evaluator: ReflexionEvaluator,
    self_reflection: ReflexionSelfReflection,
    action_executor: BaseActionExecutor,
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
        "act",
        RunnableLambda(
            partial(ReflexionNodes.act, actor=actor),
            afunc=partial(ReflexionNodes.aact, actor=actor),
        ),
    )

    builder.add_node(
        "execute_actions",
        RunnableLambda(
            partial(
                ReflexionNodes.execute_actions,
                action_executor=action_executor,
            ),
            afunc=partial(
                ReflexionNodes.aexecute_actions,
                action_executor=action_executor,
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
        "act",
        ReflexionEdges.should_continue_actor,
        {"yes": "execute_actions", "no": "evaluate"},
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
