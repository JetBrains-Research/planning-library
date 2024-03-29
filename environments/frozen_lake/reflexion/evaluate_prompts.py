from textwrap import dedent

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that judges whether the episodes of playing the Frozen Lake game ended in failure or success.",
        ),
        (
            "human",
            dedent("""
            In the Frozen Lake game, you move on a 2D grid. The end goal is to cross this grid from start 
            to finish without falling into any holes.
            You start at location (0,0) (upper left corner) of the frozen lake grid world, and the finish is located 
            at far extent of the world, i.e., (n - 1, n - 1) for n x n grid (lower right corner). 
            If you step on the hole, the game ends.
                
            You will be given a map and a sequence of intermediate steps from one episode of playing Frozen Lake game. Your goal is to evaluate whether the episode ended in success (i.e., the goal was obtained) or in failure (i.e., the player stepped into the hole or didn't reach a goal for some reason).   
            
            Take your time and comment your decision, but make sure to always output either 0 or 1 in the end, where 0 would mean 'the episode ended in failure' and 1 would mean 'the episode ended in success'. 
            Use the following format: [[number]]."""),
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

            Current map:
            {inputs}"""),
        ),
        MessagesPlaceholder("intermediate_steps"),
        (
            "human",
            dedent("""
            Pay attention to the 'terminated' and 'reward' in tools' output. 
            If 'reward' is set to 1, it means the goal has been reached. 
            If 'terminated' is set to True, it means the game has ended. 
            If the reward is not equal to 1 at that time, it means the player has fallen into a hole, hence, it is a failure.

            Your verdict (ALWAYS output either 0 or 1 in double brackets like [[number]]):"""),
        ),
    ]
)
