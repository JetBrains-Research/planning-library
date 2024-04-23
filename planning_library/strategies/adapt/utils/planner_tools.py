from textwrap import dedent
from typing import Any, Type, Optional, List, Literal
from planning_library.strategies.adapt.utils import ADaPTPlan
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from abc import ABC


class BaseADaPTPlannerTool(BaseTool, BaseModel, ABC):
    """Base tool for an ADaPT planner.

    Contains plan as a field, it won't be shown to models."""

    plan: ADaPTPlan = Field(exclude=True)

    class Config(BaseTool.Config):
        pass


class CheckPlanTool(BaseADaPTPlannerTool):
    name = "check_plan"
    description = dedent("""
        Get information about the current plan. 
        Will return the number of subtasks, their inputs and the selected mode of aggregation of their results (and/or).""")

    class CheckPlanInput(BaseModel): ...

    args_schema: Type[BaseModel] = CheckPlanInput

    def _run(self, *args: Any, **kwargs: Any) -> str:
        if len(self.plan.subtasks) == 0:
            observation = ["Currently, the plan doesn't contain any subtasks."]
        else:
            observation = [
                f"Currently, the plan contains {len(self.plan.subtasks)} subtasks."
            ]

            for i, subtask in enumerate(self.plan.subtasks):
                observation.append(f"{i + 1}. {subtask}")

        if self.plan.aggregation_mode is None:
            observation.append(
                "The current subtasks results aggregation mode is not defined yet."
            )
        else:
            observation.append(
                f"The current subtasks results aggregation mode is set to {self.plan.aggregation_mode}."
            )
        return "\n".join(observation)


class AddTaskTool(BaseADaPTPlannerTool):
    name = "add_task"
    description = dedent("""Add a new subtask to the current plan.""")

    class AddTaskInput(BaseModel):
        task_inputs: str = Field(
            description="Inputs for the current task. Should be a comprehensive natural language instruction that will be passed to the Executor agent."
        )
        task_position: Optional[int] = Field(
            description=dedent("""
                The position in the plan to insert this subtask to. 
                If not given, the task will be appended to the current plan. 
                Note that the indexing starts with 0."""),
            default=None,
        )

    args_schema: Type[BaseModel] = AddTaskInput

    def _run(
        self,
        task_inputs: str,
        task_position: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if task_position is None:
            self.plan.subtasks.append(task_inputs)
            return "Successfully added a new subtask to the end of the current plan."

        try:
            self.plan.subtasks.insert(task_position, task_inputs)
            return f"Successfully added a new subtask to position {task_position} in the current plan."
        except IndexError:
            return f"Couldn't insert task to position {task_position}. Currently, the plan has {len(self.plan.subtasks)} tasks."


class EditTaskTool(BaseADaPTPlannerTool):
    name = "edit_task"
    description = dedent(
        """Changes the formulation of the existing subtask in the current plan."""
    )

    class EditTaskInput(BaseModel):
        task_inputs: str = Field(
            description="New inputs for the task. Should be a comprehensive natural language instruction that will be passed to the Executor agent."
        )
        task_position: int = Field(
            description=dedent("""
                The position of the task to be edited in the current plan. 
                Note that the indexing starts with 0."""),
        )

    args_schema: Type[BaseModel] = EditTaskInput

    def _run(
        self, task_inputs: str, task_position: int, *args: Any, **kwargs: Any
    ) -> str:
        try:
            self.plan.subtasks[task_position] = task_inputs
            return f"Successfully edited the subtask at position {task_position} in the current plan."
        except IndexError:
            return f"Couldn't edit the subtask at position {task_position}. Currently, the plan has {len(self.plan.subtasks)} tasks."


class RemoveTaskTool(BaseADaPTPlannerTool):
    name = "remove_task"
    description = dedent("""Remove a subtask from the current plan.""")

    class RemoveTaskInput(BaseModel):
        task_position: int = Field(
            description="The position in the plan to delete the subtask from. Note that the indexing starts with 0."
        )

    args_schema: Type[BaseModel] = RemoveTaskInput

    def _run(self, task_position: int, *args: Any, **kwargs: Any) -> str:
        try:
            self.plan.subtasks.pop(task_position)
            return f"Successfully removed the subtask at position {task_position} in the current plan."
        except IndexError:
            return f"Couldn't remove the subtask at position {task_position}. Currently, the plan has {len(self.plan.subtasks)} tasks."


class DefineAggregationModeTool(BaseADaPTPlannerTool):
    name = "define_aggregation_mode"
    description = dedent("""Remove a subtask from the current plan.""")

    class DefineAggregationModeInput(BaseModel):
        aggregation_mode: Literal["and", "or"] = Field(
            description=dedent("""
            How the results of the subtasks defined in the current plan should be aggregated.
            The only allowed options are "and" and "or".
            * "and": all subtasks have to be completed successfully in order to consider the.
            * "or": all subtasks have to be completed successfully in order to consider the. 
            """)
        )

    args_schema: Type[BaseModel] = DefineAggregationModeInput

    def _run(
        self, aggregation_mode: Literal["and", "or"], *args: Any, **kwargs: Any
    ) -> str:
        self.plan.aggregation_mode = aggregation_mode
        return f"Successfully set aggregation mode to {aggregation_mode}"


def get_adapt_planner_tools(
    plan: Optional[ADaPTPlan] = None,
) -> List[BaseADaPTPlannerTool]:
    if plan is None:
        plan = ADaPTPlan(subtasks=[], aggregation_mode="and")

    return [
        CheckPlanTool(plan=plan),  # type: ignore[call-arg]
        AddTaskTool(plan=plan),  # type: ignore[call-arg]
        RemoveTaskTool(plan=plan),  # type: ignore[call-arg]
        EditTaskTool(plan=plan),  # type: ignore[call-arg]
        DefineAggregationModeTool(plan=plan),  # type: ignore[call-arg]
    ]
