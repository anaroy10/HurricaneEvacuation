## How to add a new Agent Type (dev_guide)

### creating a new agent type

* go over the Agent (agent.py) abstract base class, and understand the API that every agent will/should have.
* go over to FriendlyAgent (friendly_agent.py) and NonFriendlyAgent (non_friendly_agent.py) and see the differences.
* create a concrete class that inherits from FriendlyAgent or NonFriendlyAgent according to the desired behaviour,
this concrete class must implement an abstract method:
```python
from typing import Any

def decide(self, *args: Any, **kwargs: Any) -> Any:
    ...
```
which is the method that the simulator will use for running agents and will expect from them to return a
decision regarding the next in game move.

* Example:

```python
import networkx as nx
from networkx import Graph
from typing import override, Any

from SearchAgents.agents.friendly_agent import FriendlyAgent
from SearchAgents.consts import NUM_PEOPLE

class TeleportAgent(FriendlyAgent):
    """ agent that teleports to a given destination node and picks up the people there """

    def __init__(self, g: Graph, initial_node: int) -> None:
        """ inits a new teleport agent """
        super().__init__(g, initial_node)

    @override
    def decide(self, *args: Any, **kwargs: Any) -> Any:
        """ decision strategy is to jump to the node with most people """
        max_node = max(self._world.nodes(data=True), key=lambda x: x[1].get(NUM_PEOPLE, 0))
        node_id, node_data = max_node

        # if all nodes has 0 people
        if node_data.get(NUM_PEOPLE, 0) == 0:
            return "terminate"

        # if exists a node with `> 0` num_people, jump to it
        return f"traverse {node_id}"

    @override
    def traverse(self, dst_node: int, zero_weight: bool = False) -> None:
        """ overriding traverse because this agent jumps, so the logic is not traversing by edges """
        dst_node_data: dict[str, int] = self._world.nodes[dst_node]

        people_to_save: int = dst_node_data.get(NUM_PEOPLE, 0)

        self._current_node = dst_node
        self._people_saved += people_to_save
        self._score += 1000 * people_to_save

        nx.set_node_attributes(self._world, {dst_node: {NUM_PEOPLE: 0}})
```
* in that example we've seen how to add a fully new supported agent to the environment, in that way we can add any
agent we'll want.
* the idea is that the `decide()` method, will do the needed calculations (e.g., Dijkstra/A* ...) and update internal
states, then return the next move to make.
* if needed, override any other method provided by Agent base class, to adjust behaviour as you like.

### updating simulation of new type
* after new type has been created, we should import it inside the `__main__.py`.
* then, we need to add the type into the agents map:
```python
# the else needed types for this example
from SearchAgents.agents.agent import Agent
from SearchAgents.agents.human_agent import HumanAgent
from SearchAgents.agents.stupid_greedy_agent import StupidGreedyAgent
from SearchAgents.agents.thief_agent import ThiefAgent

# the new type we've added previously
from SearchAgents.agents.teleport_agent import TeleportAgent

# one source of truth for all agent types
agents_map: dict[str, type[Agent]] = {
    "human": HumanAgent,
    "greedy": StupidGreedyAgent,
    "thief": ThiefAgent,
    "teleport": TeleportAgent  # <-- added here a new option (now, the simulator will prompt you for it)
}
```
* because of the Agent's API generality, else is automatically handled :)
