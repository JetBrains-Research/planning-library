# 🤖📝 Planning Library

## Installation

> :construction: TODO

## Quick Tour

> :construction: Subject to change

Currently, we have two types of strategies: **custom strategies** and *
*[LangGraph](https://github.com/langchain-ai/langgraph/tree/main) strategies**.

### Custom strategies

Custom strategies follow the interface provided
by [`BaseStrategy`](planning_library/custom_agent_executor/strategies/base_strategy.py). They are designed to be used in
a very similar way to the
default [`AgentExecutor`](https://python.langchain.com/docs/modules/agents/quick_start#create-the-agent) from LangChain.

**Example:** [Tree of Thoughts + DFS](planning_library/custom_agent_executor/strategies/tot_strategy/tot_strategy.py).

### LangGraph strategies

Strategies powered by [LangGraph library](https://github.com/langchain-ai/langgraph) provide builder methods that
require an agent and tools and return a
compiled [`StateGraph`](https://github.com/langchain-ai/langgraph?tab=readme-ov-file#stategraph) that exposes the same
interface as any LangChain runnable.

**Example:** [Reflexion](planning_library/langgraph_version/strategies/reflexion/reflexion.py).

## Available Strategies

|                     Name                      |       Implementation        |   Type    |                                                Paper                                                 |
|:---------------------------------------------:|:---------------------------:|:---------:|:----------------------------------------------------------------------------------------------------:|
| :construction: Tree of Thoughts + DFS / DFSDT | `TreeOfThoughtsDFSStrategy` |  Custom   | [:scroll: ToT](https://arxiv.org/abs/2305.10601), [:scroll: DFSDT](https://arxiv.org/abs/2307.16789) |
|   :construction:                  Reflexion   | `create_reflexion_strategy` | LangGraph |                             [:scroll:](https://arxiv.org/abs/2303.11366)                             |

## Available Environments

---

1. :construction: Game of 24

   Game of 24 is a mathematical reasoning task. The goal is to reach the number 24 by applying arithmetical operations
   to four given numbers. See :scroll: [Tree of Thoughts](https://arxiv.org/abs/2305.10601) paper for more details.

   Our implementation of Game of 24 is available under [`environments/game_of_24`](environments/game_of_24) folder. It
   includes a set of prompts, a set of tools and examples of running available strategies on Game of 24.