from itertools import combinations
from typing import override, Dict

import networkx as nx

from SearchAgents.consts import KIT_SLOWER, WEIGHT
from SearchAgents.types.world_state import WorldState
from SearchAgents.heuristics.heuristic_strategy import HeuristicStrategy


class HurricaneEvacuationHeuristic(HeuristicStrategy):
    """ Computes a MST-based admissible heuristic for the hurricane evacuation problem """
    def __init__(self, world_graph: nx.Graph):
        """
        Build a RELAXED graph and precompute all-pairs shortest paths

        Notes:
            - the relaxed graph built with the assumption the agent already has a kit and can traverse anywhere
            - it is admissible because ignores the equip and unequip operation costs

        Args:
            world_graph (nx.Graph): the world graph where the agent lives in the simulation
        """
        super().__init__(world_graph)

        self._relaxed_graph: nx.Graph = self.__build_relaxed_graph()
        self._all_shortest_dists: Dict[int, Dict[int, int]] = self.__compute_all_pairs_shortest_paths()

    def __build_relaxed_graph(self) -> nx.Graph:
        """
        relaxes the environment graph - treating all edges as unflooded

        Returns:
            nx.Graph: a relaxed graph (deep copied)
        """

        relaxed_graph: nx.Graph = nx.Graph()
        for u, v, data in self._world.edges(data=True):
            edge_weight: int = data.get(WEIGHT, 1)
            relaxed_graph.add_edge(u, v, weight=edge_weight)

        return relaxed_graph

    def __compute_all_pairs_shortest_paths(self) -> Dict[int, Dict[int, int]]:
        """ computing all pairs shortest paths (shortest paths from every node to every other node) """
        return dict(
            nx.all_pairs_dijkstra_path_length(self._relaxed_graph)
        )

    def __build_complete_graph(self, clique_nodes: list[int]) -> nx.Graph:
        """
        builds a complete graph of shortest paths

        Args:
            clique_nodes (list[int]): current node + goal nodes to visit

        Returns:
            nx.Graph: a clique graph of all nodes, connected via shortest paths
        """
        clique_edges: list[tuple[int, int, int]] = [
            (u, v, self._all_shortest_dists[u].get(v, 1)) for u, v in combinations(clique_nodes, 2)
        ]

        clique: nx.Graph = nx.Graph()
        clique.add_weighted_edges_from(clique_edges)

        return clique

    @override
    def estimate(self, state: WorldState) -> int:
        """
        finds MST over the shortest paths complete graph of { current_node } && remaining_targets
        then calculate sum of weights

        Args:
            state (WorldState): a state the including the current node and remaining goal nodes

        Returns:
            int: the heuristics score of the current state
        """
        # unpacking state
        agent_location: int = state.agent_location
        people_locations: list[int] = list(state.people_locations)

        # if we at goal target, h(s) := 0
        if not people_locations:
            return 0

        clique: nx.Graph = self.__build_complete_graph([agent_location] + people_locations)

        clique_mst: nx.Graph = nx.minimum_spanning_tree(clique, algorithm="prim")

        # h(s) := SUM_{e in MST}(w(e))
        return sum(
            data.get(WEIGHT, 1) for *_, data in clique_mst.edges(data=True)
        )
