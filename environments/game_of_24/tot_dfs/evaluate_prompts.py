from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

evaluate_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that judges whether answers to Game of 24 are correct or individual steps to reach 24 from given numbers seem plausible.",
        ),
        (
            "human",
            "You are given inputs and intermediate steps for Game of 24: the goal is to reach 24 via arithmetical operations on given numbers. In your case, Game of 24 is played in step-by-step fashion: each suggestion is an arithmetical operation between two numbers. You might be given either a suggestion (a single arithmetical operation on two numbers) or a final answer (mathematical expression that should be equal to 24). In the former case, evaluate if a new suggestion is correct and how likely it is to help in reaching 24. In the latter case, evaluate if the final answer is correct, i.e., uses each input number exactly once. You are allowed to comment your decision, but make sure to always output one of the following words in the end: 'sure', 'likely', 'impossible'.",
        ),
        ("human", "Input: {inputs}"),
        MessagesPlaceholder("trajectory"),
        ("ai", "Suggestion: {next_thought} = {observation}"),
        ("human", "Judge:"),
    ]
)
