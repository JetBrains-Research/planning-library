from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from textwrap import dedent

plan_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an advanced reasoning agent that can create plans for reasoning tasks.",
        ),
        (
            "human",
            dedent("""
            The task at hand is related to a household navigation. An attempt to solve it directly has failed, and you need to create a clear and concise plan instead.
            
            You will be given the task description, all the intermediate steps that were taken during the last trial and the final outcome. 
            
            Task description:
            {inputs}"""),
        ),
        ("human", "Intermediate steps:"),
        MessagesPlaceholder("intermediate_steps"),
        ("human", "Final outcome: {agent_outcome}"),
        (
            "human",
            dedent("""
        Please, output a step-by-step plan for how to solve the given task successfully. 
        Take note of the available tools for the current environment, don't suggest doing something that wouldn't be possible.
        Do not use complicated logic in your plan: all the steps should be executed sequentially, with no 'or' or 'if'.
        
        Each step should be a self-contained instruction for the corresponding subtask. 
        If there are useful observations included in the task formulation or encountered during the trial (like a list of objects present in a room or a list of objects located in a certain place), do not hesitate to include them in a step.
        
        Use the following format:
        
        Step 1: <Description of first step>
        Step 2: <Description of second step>
        ...
        Step N: <Description of Nth step>
        
        Here is an examplee:
        
        You are in the middle of a room. Looking quickly around you, you see a desk 1, microwave 1, a cabinet 3, a cabinet 9, a
        drawer 2, a coffeemachine 1, a stoveburner 4, a drawer 5, a cabinet 11, a drawer 3, a stoveburner 1, a drawer 1, a
        toaster 1, a fridge 1, a stoveburner 2, a cabinet 6, a cabinet 10, a countertop 1, a cabinet 13, a cabinet 7, a
        garbagecan 1, a cabinet 2, a cabinet 8, a cabinet 12, a drawer 4, a cabinet 1, a sinkbasin 1, a cabinet 5, a
        stoveburner 3, and a cabinet 4.
        
        Goal: Put a mug in/on desk.
        
        Output:
        
        # Think: To perform this task, I need to find and take mug and then put it on desk. First, I will focus on finding mug. 
        # Think: Now that I am carrying mug, I will focus on putting it in/on desk.
        
        Step 1: You are in the middle of a room. Looking quickly around you, you see a desk 1, microwave 1, a cabinet 3, a cabinet 9, a drawer 2, a coffeemachine 1, a stoveburner 4, a drawer 5, a cabinet 11, a drawer 3, a stoveburner 1, a drawer 1, a toaster 1, a fridge 1, a stoveburner 2, a cabinet 6, a cabinet 10, a countertop 1, a cabinet 13, a cabinet 7, a garbagecan 1, a cabinet 2, a cabinet 8, a cabinet 12, a drawer 4, a cabinet 1, a sinkbasin 1, a cabinet 5, a stoveburner 3, and a cabinet 4. Find and take mug
        Step 2: You are in the middle of a room. Looking quickly around you, you see a desk 1, microwave 1, a cabinet 3, a cabinet 9, a drawer 2, a coffeemachine 1, a stoveburner 4, a drawer 5, a cabinet 11, a drawer 3, a stoveburner 1, a drawer 1, a toaster 1, a fridge 1, a stoveburner 2, a cabinet 6, a cabinet 10, a countertop 1, a cabinet 13, a cabinet 7, a garbagecan 1, a cabinet 2, a cabinet 8, a cabinet 12, a drawer 4, a cabinet 1, a sinkbasin 1, a cabinet 5, a stoveburner 3, and a cabinet 4. Put mug in/on desk
        """),
        ),
    ]
)
