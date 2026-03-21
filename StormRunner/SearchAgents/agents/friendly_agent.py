from abc import ABC
from typing import override, Dict

import networkx as nx

from SearchAgents.agents.agent import Agent
from SearchAgents.consts import KIT_SLOWER, NUM_PEOPLE, WEIGHT


class FriendlyAgent(Agent):
    """
    defines an agent which will pick up people in the game

    Notes:
        currently, its every agent except the ThiefAgent (which is non-friendly)
    """
    def __init__(self, g: nx.Graph, initial_node: int) -> None:
        super().__init__(g, initial_node)

        # if the agent starts on a node with people, take them immediately
        initial_node_attr = self._world.nodes[initial_node]
        if initial_node_attr.get(NUM_PEOPLE, 0):
            self.traverse(self._initial_node, zero_weight=True)

    @override
    def traverse(self, dst_node: int, zero_weight: bool = False) -> None:
        """
        1. moves the agent to a new node
        2. rescues all the people on the new node (friendly behaviour)
        3. calculate new score (accordingly to the amount of people and edge's weight)
        """
        weight_to_add: int = 0 if zero_weight \
            else self._world.get_edge_data(self._current_node, dst_node).get(WEIGHT, 1)
        dst_node_data: Dict[str, int] = self._world.nodes[dst_node]

        time_to_add: int = weight_to_add
        kit_slower: int = self._world.graph.get(KIT_SLOWER, 1) if self._is_hold_kit else 1
        people_to_save: int = dst_node_data.get(NUM_PEOPLE, 0)
        time_cost: int = time_to_add * kit_slower

        self._current_node = dst_node
        self._total_time += time_cost
        self._people_saved += people_to_save
        self._score += 1000 * people_to_save - time_cost

        nx.set_node_attributes(self._world, {dst_node: {NUM_PEOPLE: 0}})
