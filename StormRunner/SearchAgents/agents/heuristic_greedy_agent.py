from typing import Any, Tuple, List, Callable, override

import networkx as nx

from SearchAgents.agents.heuristic_agent import HeuristicAgent
from SearchAgents.consts import WEIGHT, FLOODED, HAS_KIT
from SearchAgents.types.world_state import WorldState
from SearchAgents.utils.utils import get_target_nodes
from SearchAgents.heuristics.hurricane_evacuation_heuristic import HurricaneEvacuationHeuristic


class HeuristicGreedyAgent(HeuristicAgent):
    """
    Heuristics-Greedy search agent
    At each step, picks the move with the best immediate heuristic value to expand next.
    """
    def __init__(
            self, g: nx.Graph, initial_node: int,
            heuristic_strategy: HurricaneEvacuationHeuristic, expansion_limit: int, time_per_expansion: float
    ) -> None:
        """ inits a heuristics greedy search agent """
        super().__init__(g, initial_node, heuristic_strategy, expansion_limit, time_per_expansion)

    def __get_heuristic_cost(self, current_node: int, remaining: list[int]) -> int:
        """ returns the heuristic cost from the provided heuristic strategy """
        return self._heuristic_strategy.estimate(
            WorldState(
                current_node,
                frozenset(remaining)
            )
        )

    @override
    def decide(self) -> str:
        target_nodes: list[int] = get_target_nodes(self._world)

        if not target_nodes:
            return "terminate"

        candidates: List[Tuple[float, str]] = []

        neighbors = sorted(self._world.neighbors(self._current_node))

        # traverse
        for nbr in neighbors:
            edge = self._world.get_edge_data(self._current_node, nbr) or {}
            w = int(edge.get(WEIGHT, 1))
            flooded = bool(edge.get(FLOODED, False))

            if flooded and not self._is_hold_kit:
                continue

            next_node = nbr
            remaining = [t for t in target_nodes if t != next_node]
            h_future = self.__get_heuristic_cost(next_node, remaining)

            candidates.append((h_future, f"traverse {nbr}"))

        remaining = list(target_nodes)

        # equip
        if self._world.nodes[self._current_node].get(HAS_KIT, False) and not self._is_hold_kit:
            h_future = self.__get_heuristic_cost(self._current_node, remaining)
            candidates.append((h_future, "equip"))

        # unequip
        if self._is_hold_kit:
            h_future = self.__get_heuristic_cost(self._current_node, remaining)
            candidates.append((h_future, "unequip"))

        # no-op
        if not candidates:
            return "no-op"

        # pick best action
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
