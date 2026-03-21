import unittest
from typing import FrozenSet

import networkx as nx

from SearchAgents.heuristics.hurricane_evacuation_heuristic import HurricaneEvacuationHeuristic
from SearchAgents.types.world_state import WorldState


class TestEvacuationHeuristic(unittest.TestCase):
    def setUp(self):
        g = nx.Graph()

        g.graph["kit_slower"] = 2
        g.graph["equip_time"] = 1
        g.graph["unequip_time"] = 1

        # num_people marks remaining targets
        g.add_node(0, num_people=0)
        g.add_node(1, num_people=1)
        g.add_node(2, num_people=0)
        g.add_node(3, num_people=0)
        g.add_node(4, num_people=1)

        g.add_edge(0, 1, weight=2, flooded=True)
        g.add_edge(0, 2, weight=5)
        g.add_edge(1, 3, weight=3)
        g.add_edge(2, 3, weight=2)
        g.add_edge(3, 4, weight=2)

        self.graph = g
        self.heuristic = HurricaneEvacuationHeuristic(self.graph)

    def test_relaxed_graph_edges(self):
        """ checks that weights not broke when creating the relaxed graph """
        expected_edges: dict[tuple[int, int], int] = {
            (0, 1): 2,
            (0, 2): 5,
            (1, 3): 3,
            (2, 3): 2,
            (3, 4): 2
        }

        for u, v, d in self.heuristic._relaxed_graph.edges(data=True):
            self.assertEqual(
                d["weight"],
                expected_edges[(u, v)] if (u, v) in expected_edges else expected_edges[(v, u)]
            )

    def test_mst_heuristic_values(self):
        """ test heuristic computation for several states """
        test_cases: list[tuple[int, list[int], int]] = [
            (0, [1, 4], 7),
            (1, [1, 4], 5),
            (2, [1, 4], 9),
            (3, [1, 4], 5),
            (4, [1, 4], 5),
            (0, [], 0)
        ]

        for current, remaining, expected in test_cases:
            h_val = self.heuristic.estimate(WorldState(current, frozenset(remaining)))
            self.assertEqual(h_val, expected)
