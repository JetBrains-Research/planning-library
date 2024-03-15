from textwrap import dedent

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

generate_openai_tools_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an intelligent agent playing a Frozen Lake game."),
        (
            "human",
            dedent(
                """
            In Frozen Lake game, you move on a 2D grid. Your goal is to cross this grid from start 
            to finish without falling into any holes.
            
            You start at location (0,0) (upper left corner) of the frozen lake grid world, and the finish is located 
            at far extent of the world, i.e., (n - 1, n - 1) for n x n grid (lower right corner).
            
            The first coordinate is X axis.
            When you move left, you decrease your first coordinate by 1 (can't be bigger than n - 1).
            When you move right, you increase your first coordinate by 1 (can't be lower than 0).
            
            The second coordinate is Y axis. 
            When you move down, you increase your second coordinate by 1 (can't be bigger than n - 1).
            When you move right, you decrease your second coordinate by 1 (can't be lower than 0).
            
            The map is an n x n grid where different types of cells are denoted by different letters:
            * S - start cell
            * G - goal cell
            * F - frozen cell
            * H - hole cell

            Example for 2 x 2 case:
            SH
            FG
            
            Use tools to move; refrain from calling tools only when the game has ended. Pay attention to the `terminated`
            flag returned by the tool calls: if it is set to True, it means the game has ended, you won't be allowed to move 
            anymore.
            
            Current map:
            {inputs} 
            """
            ),
        ),
        MessagesPlaceholder("agent_scratchpad"),
        (
            "human",
            "You might have already made some suggestions for the current state - if you did, you will find them below.",
        ),
        MessagesPlaceholder("previous_thoughts"),
        (
            "human",
            "Please, suggest exactly ONE (1) tool call, no more and no less, DIFFERENT from your previous suggestions.",
        ),
    ]
)
