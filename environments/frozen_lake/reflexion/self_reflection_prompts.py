from textwrap import dedent

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import BaseOutputParser

self_reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an advanced reasoning agent that can self-reflect on their shortcomings when solving reasoning tasks.",
        ),
        (
            "human",
            dedent("""
            You will be given your previous trial in the Frozen Lake game.
            
            In Frozen Lake game, you move on a 2D grid. The end goal is to cross this grid from start 
            to finish without falling into any holes.

            You start at location (0,0) (upper left corner) of the frozen lake grid world, and the finish is located 
            at far extent of the world, i.e., (n - 1, n - 1) for n x n grid (lower right corner). 
            
            The first coordinate is X axis. 
            When you move right, you increase your first coordinate by 1 (can't be bigger than n - 1).
            When you move left, you decrease your first coordinate by 1 (can't be lower than 0).
            
            The second coordinate is Y axis.
            When you move up, you decrease your second coordinate by 1 (can't be lower than 0).
            When you move down, you increase your second coordinate by 1 (can't be bigger than n - 1).
            
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
        MessagesPlaceholder("intermediate_steps"),
        (
            "human",
            dedent("""
        In this trial, you were unsuccessful. 
        In a few sentences, diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same shortcomings. 
        Use complete sentences."""),
        ),
    ]
)


class FrozenLakeSelfReflexionOutputParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        return text
