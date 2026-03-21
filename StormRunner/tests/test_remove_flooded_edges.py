import unittest
import networkx as nx
from SearchAgents.utils.utils import remove_flooded_edges


class TestRemoveFloodedEdges(unittest.TestCase):
    def test_removes_only_flooded_edges(self):
        g = nx.Graph()
        g.add_edge("A", "B", flooded=True)
        g.add_edge("B", "C", flooded=False)
        g.add_edge("C", "D", flooded=False)

        result = remove_flooded_edges(g)

        self.assertNotIn(("A", "B"), result.edges)
        self.assertIn(("B", "C"), result.edges)
        self.assertIn(("C", "D"), result.edges)

    def test_original_graph_is_not_modified(self):
        g = nx.Graph()
        g.add_edge(1, 2, flooded=True)

        _ = remove_flooded_edges(g)

        self.assertIn((1, 2), g.edges)

    def test_no_flooded_edges(self):
        g = nx.Graph()
        g.add_edge("X", "Y")
        g.add_edge("Y", "Z")

        result = remove_flooded_edges(g)

        self.assertEqual(set(result.edges), set(g.edges))

    def test_all_edges_flooded(self):
        g = nx.Graph()
        g.add_edge("A", "B", flooded=True)
        g.add_edge("B", "C", flooded=True)

        result = remove_flooded_edges(g)

        self.assertEqual(len(result.edges), 0)

    def test_returns_a_copy_not_same_object(self):
        g = nx.Graph()
        g.add_edge(1, 2, flooded=True)

        result = remove_flooded_edges(g)

        self.assertIsNot(result, g)
        self.assertIsNot(result.nodes, g.nodes)
        self.assertIsNot(result.edges, g.edges)
