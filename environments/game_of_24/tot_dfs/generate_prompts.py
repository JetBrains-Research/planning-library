from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

generate_propose_openai_tools_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that plays Game of 24."),
        (
            "human",
            "Your end goal is to obtain 24 from given numbers via basic arithmetic operations with given numbers. Use {max_num_thoughts} of available tools to suggest possible next step(s) from the current state. Make sure to suggest exactly {max_num_thoughts} tool call(s), no more and no less.\nInputs: {inputs}",
        ),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

generate_sample_openai_tools_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that plays Game of 24."),
        (
            "human",
            "Your end goal is to obtain 24 from given numbers via basic arithmetic operations. Let's play Game of 24 in a step-by-step fashion: use only one of available tools to suggest a possible next step from the current state. Please, make sure to suggest exactly ONE (1) tool call, no more and no less. Refrain from calling tools only when you're ready to give a final answer. In this case, make sure to include a mathematical expression showing how to obtain 24 from given numbers, for instance: '(2 + 2) * (12 / 2) = 24'\nInputs: {inputs}",
        ),
        MessagesPlaceholder("agent_scratchpad"),
        (
            "human",
            "You might have already made some suggestions for the current state - if you did, you will find them below. Don't repeat yourself.",
        ),
        MessagesPlaceholder("previous_thoughts"),
    ]
)
