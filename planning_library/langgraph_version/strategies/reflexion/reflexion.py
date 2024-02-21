from typing import Literal

from langchain_core.agents import AgentFinish
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph  # type: ignore[import]

from planning_library.langgraph_version.strategies.reflexion.actors import BaseActor
from planning_library.langgraph_version.strategies.reflexion.evaluators import BaseEvaluator
from planning_library.langgraph_version.strategies.reflexion.self_reflections import BaseSelfReflection
from planning_library.langgraph_version.utils import AgentState


def create_reflexion_strategy(
    actor: BaseActor, evaluator: BaseEvaluator, self_reflection: BaseSelfReflection, max_num_iterations: int
):
    def _should_continue_actor(state: AgentState) -> Literal["yes", "no"]:
        return "no" if isinstance(state["agent_outcome"], AgentFinish) else "yes"

    def _should_continue_num_iterations(state: AgentState) -> Literal["yes", "no"]:
        if state["iteration"] >= max_num_iterations:
            return "no"
        return "yes"

    def _init_state(state: AgentState) -> AgentState:
        return {
            "inputs": state["inputs"],
            "agent_outcome": None,
            "evaluator_score": None,
            "self_reflections": [],
            "intermediate_steps": [],
            "iteration": 1,
        }

    builder = StateGraph(AgentState)
    builder.add_node("inite_state", _init_state)
    builder.add_node("act", RunnableLambda(actor.act, afunc=actor.aact))
    builder.add_node("execute_tools", RunnableLambda(actor.execute_tools, afunc=actor.aexecute_tools))
    builder.add_node("evaluate", RunnableLambda(evaluator.evaluate, afunc=evaluator.aevaluate))
    builder.add_node("self_reflect", RunnableLambda(self_reflection.self_reflect, afunc=self_reflection.aself_reflect))

    builder.set_entry_point("inite_state")
    builder.add_edge("inite_state", "act")

    builder.add_conditional_edges("act", _should_continue_actor, {"yes": "execute_tools", "no": "evaluate"})
    builder.add_edge("execute_tools", "act")
    builder.add_conditional_edges(
        "evaluate",
        evaluator.should_continue,
        {
            "yes": "self_reflect",
            "no": END,
        },
    )
    builder.add_conditional_edges("self_reflect", _should_continue_num_iterations, {"yes": "act", "no": END})
    return builder.compile()
