# 🤖📝 Planning Library

## Installation

> :construction: TODO

## Quick Tour

> :construction: Subject to change

Currently, we have two types of strategies: **custom strategies** and
**[LangGraph](https://github.com/langchain-ai/langgraph/tree/main) strategies**.

### Custom strategies

Custom strategies follow the interface provided
by [`BaseCustomStrategy`](planning_library/strategies/base_strategy.py).

**Example:** [Tree of Thoughts + DFS](planning_library/strategies/tot_dfs/tot_strategy.py)

#### Initializing strategy

Each custom strategy can be created by envoking a static method `create` with at least agent and tools.

```python
from planning_library.strategies import TreeOfThoughtsDFSStrategy

agent = ...  # any runnable that follows either RunnableAgent or RunnableMultiActionAgent
tools = [...]  # any sequence of tools
strategy_executor = TreeOfThoughtsDFSStrategy.create(
    agent=agent,
    tools=tools,
)
```

Some strategies contain other meaningful components (e.g., an evaluator, which is responsible for evaluating
intermediate steps). We (:construction: will) provide some default implementations for them, but feel free to redefine
them with custom runnables tailored for specific tasks.

#### Using strategy

Each custom strategy is an instance of [`Chain`](https://python.langchain.com/docs/modules/chains/) and mostly can be
used the same
way as the default [`AgentExecutor`](https://python.langchain.com/docs/modules/agents/quick_start#create-the-agent) from
LangChain.

```
strategy_executor.invoke({"inputs": "Hello World"})
```

### LangGraph strategies

Strategies powered by [LangGraph library](https://github.com/langchain-ai/langgraph) follow the interface provided
by [`BaseLangGraphStrategy`](planning_library/strategies/base_strategy.py).

**Example:** [Reflexion](planning_library/strategies/reflexion/reflexion_strategy.py)

#### Initializing strategy

Each LangGraph strategy can be created by envoking a static method `create` with (at least) agent and tools.

```python
from planning_library.strategies import ReflexionStrategy

agent = ...  # any runnable that follows either RunnableAgent or RunnableMultiActionAgent
tools = [...]  # any sequence of tools
strategy_graph = ReflexionStrategy.create(agent=agent, tools=tools)
```

Some strategies contain other meaningful components (e.g., an evaluator, which is responsible for evaluating
intermediate steps). We (:construction: will) provide some default implementations for them, but feel free to redefine
them with custom runnables tailored for specific tasks.

#### Using strategy

[`BaseLangGraphStrategy.create`](planning_library/strategies/base_strategy.py) returns a
compiled [`StateGraph`](https://github.com/langchain-ai/langgraph?tab=readme-ov-file#stategraph) that exposes the same
interface as any LangChain runnable.

```python
strategy_graph.invoke({"inputs": "Hello World"})
```

## Available Strategies

|                     Name                      |                                   Implementation                                   |   Type    |                                                Paper                                                 |
|:---------------------------------------------:|:----------------------------------------------------------------------------------:|:---------:|:----------------------------------------------------------------------------------------------------:|
| :construction: Tree of Thoughts + DFS / DFSDT | [`TreeOfThoughtsDFSStrategy`](planning_library/strategies/tot_dfs/tot_strategy.py) |  Custom   | [:scroll: ToT](https://arxiv.org/abs/2305.10601), [:scroll: DFSDT](https://arxiv.org/abs/2307.16789) |
|   :construction:                  Reflexion   | [`ReflexionStrategy`](planning_library/strategies/reflexion/reflexion_strategy.py) | LangGraph |                             [:scroll:](https://arxiv.org/abs/2303.11366)                             |

## Available Environments

---

1. :construction: Game of 24

   Game of 24 is a mathematical reasoning task. The goal is to reach the number 24 by applying arithmetical operations
   to four given numbers. See :scroll: [Tree of Thoughts](https://arxiv.org/abs/2305.10601) paper for more details.

   Our implementation of Game of 24 is available under [`environments/game_of_24`](environments/game_of_24) folder. It
   includes a set of prompts, a set of tools and examples of running available strategies on Game of 24.