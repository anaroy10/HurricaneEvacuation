from typing import override

import networkx as nx

from SearchAgents.agents.friendly_agent import FriendlyAgent
from SearchAgents.consts import NUM_PEOPLE
from SearchAgents.utils.utils import remove_flooded_edges
from SearchAgents.utils.utils import get_target_nodes


class StupidGreedyAgent(FriendlyAgent):
    """ greedy agent that mainly focus on shortest paths """
    def __init__(self, g: nx.Graph, initial_node: int) -> None:
        """ inits a greedy agent """
        super().__init__(g, initial_node)

        # as an internal state, the greedy agent stores copy of the graph, but without flooded edges
        # it is stupid, so it do not think of equipping amphibian kits => it should not take them in count of shortest path
        self.unflooded_world: nx.Graph = remove_flooded_edges(self._world)

    @override
    def decide(self) -> str:
        """
        stupid-greedy strategy:
            1. find the node with most people on
            2. calculate the shortest path that does not traverse any flooded edge
            3. start going in that path, until node is reached

        Notes:
            1. we calculate which nodes has people every round, why:
                a. it is a stupid agent
                b. other agents might have changed the state of the graph

        Returns:
            traverse to next node in the shortest path or terminate if there are no more nodes with people to rescue
        """
        nodes_with_people: list[int] = get_target_nodes(self._world)

        if not nodes_with_people:
            return "terminate"

        target_node = max(nodes_with_people, key=lambda n: (self._world.nodes[n].get(NUM_PEOPLE, 0), -n))

        try:
            path = nx.shortest_path(self.unflooded_world, self._current_node, target_node, weight="weight")
            return f"traverse {path[1]}"
        except nx.NetworkXNoPath:
            return "terminate"
