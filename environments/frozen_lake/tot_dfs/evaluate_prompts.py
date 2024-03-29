from textwrap import dedent

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that judges whether suggestions for individual steps in Frozen Lake game seem plausible.",
        ),
        (
            "human",
            dedent("""
            In Frozen Lake game, you move on a 2D grid. The end goal is to cross this grid from start 
            to finish without falling into any holes.

            You start at location (0,0) (upper left corner) of the frozen lake grid world, and the finish is located 
            at far extent of the world, i.e., (n - 1, n - 1) for n x n grid (lower right corner). 
            
            The first coordinate is Y axis.
            When you move up, you decrease your first coordinate by 1 (can't be bigger than n - 1).
            When you move down, you increase your first coordinate by 1 (can't be lower than 0).
            
            The second coordinate is X axis. 
            When you move right, you increase your second coordinate by 1 (can't be bigger than n - 1).
            When you move left, you decrease your second coordinate by 1 (can't be lower than 0).
            
            You will be given a map, optionally, a sequence of intermediate steps and a suggestion for the next step. Your goal is
            to evaluate whether this new suggestion brings one closer to achieving the goal or not.   
            
            Take your time and comment your decision, but make sure to always output a number between 0 and 1 in the end, where 0 would mean 'current suggestion is incorrect or makes reaching the goal impossible' and 1 would mean 'current suggestion will surely help in achieving the goal'. 
            Use the following format: [[number]].
            """),
        ),
        (
            "human",
            dedent("""
            The map is an n x n grid where different types of cells are denoted by different letters:
            * S - start cell
            * G - goal cell
            * F - frozen cell
            * H - hole cell
            
            Example for 2 x 2 case:
            SH
            FG
            
            If you step on the hole, the game ends. You are allowed to step on the frozen cells, but note that they are slippery, so there is a probability that you will move perpendicular to the intended direction.
            
            Current map:
            {inputs}"""),
        ),
        MessagesPlaceholder("trajectory"),
        ("human", "Suggestion for the next step: {next_thought}"),
        (
            "human",
            dedent("""Pay attention to the 'terminated' in tools' output, if it is set to True, it means the game has ended, output [[0]] for all suggestions in this case.
            Do not be overly strict, only reject the moves that are likely to lead to a hole.
            Your verdict about the suggestion for the next step (ALWAYS output a number between 0 and 1 in the end in a format [[number]]):"""),
        ),
    ]
)
