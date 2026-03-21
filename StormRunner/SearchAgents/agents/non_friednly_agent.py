from abc import ABC
from typing import override, Dict

import networkx as nx

from SearchAgents.agents.agent import Agent
from SearchAgents.consts import WEIGHT, KIT_SLOWER


class NonFriendlyAgent(Agent):
    """ defines an agent which not picks up any people in the game """
    def __init__(self, g: nx.Graph, initial_node: int) -> None:
        super().__init__(g, initial_node)

    @override
    def traverse(self, dst_node: int, zero_weight: bool = False) -> None:
        """ not picks up any people while traversing to new nodes (non-friendly behaviour) """
        weight_to_add: int = self._world.get_edge_data(self._current_node, dst_node).get(WEIGHT, 1)

        time_to_add: int = weight_to_add
        kit_slower: int = self._world.graph.get(KIT_SLOWER, 1) if self._is_hold_kit else 1
        time_cost: int = time_to_add * kit_slower

        self._current_node = dst_node
        self._total_time += time_cost
        self._score -= time_cost
