import unittest
from pathlib import Path
from networkx import Graph

from SearchAgents.utils.hurricane_graph_parser import HurricaneGraphParser


class TestHurricaneGraphParser(unittest.TestCase):
    """ test suite to test that the logic of graph generation """
    @classmethod
    def setUpClass(cls) -> None:
        """ adds a property of test cases directory, to the class """
        cls.graphs_dir  = Path(__file__).parent / "graphs"

    def util_assert_graph(self, g: Graph, expected_graph, expected_nodes, expected_edges) -> None:
        self.assertEqual(g.graph, expected_graph)
        self.assertEqual(list(g.nodes(data=True)), expected_nodes)
        self.assertEqual(list(g.edges(data=True)), expected_edges)

    def test_graph_assignment_example(self) -> None:
        file_path: Path = self.graphs_dir / "graph_assignment_example.txt"
        g: Graph = HurricaneGraphParser(file_path).get_hurricane_graph()

        expected_graph = {'undirected': True, 'equip_time': 2, 'unequip_time': 1, 'kit_slower': 3}
        expected_nodes = [(1, {'has_kit': True, 'num_people': 0}), (2, {'has_kit': False, 'num_people': 1}),
                          (3, {'has_kit': False, 'num_people': 0}), (4, {'has_kit': True, 'num_people': 2})]
        expected_edges = [(1, 2, {'edge_id': 1, 'weight': 1, 'flooded': True}),
                          (1, 3, {'edge_id': 4, 'weight': 4, 'flooded': False}),
                          (2, 3, {'edge_id': 3, 'weight': 1, 'flooded': False}),
                          (2, 4, {'edge_id': 5, 'weight': 5, 'flooded': False}),
                          (3, 4, {'edge_id': 2, 'weight': 1, 'flooded': False})]

        self.util_assert_graph(g, expected_graph, expected_nodes, expected_edges)

    def test_graph_all_edges_flooded(self) -> None:
        file_path: Path = self.graphs_dir / "graph_all_edges_flooded.txt"
        g: Graph = HurricaneGraphParser(file_path).get_hurricane_graph()

        expected_graph = {'undirected': True, 'equip_time': 1, 'unequip_time': 1, 'kit_slower': 2}
        expected_nodes = [(1, {'has_kit': True, 'num_people': 0}), (2, {'has_kit': False, 'num_people': 1}),
                          (3, {'has_kit': False, 'num_people': 0})]
        expected_edges = [(1, 2, {'edge_id': 1, 'weight': 1, 'flooded': True}),
                          (1, 3, {'edge_id': 3, 'weight': 3, 'flooded': True}),
                          (2, 3, {'edge_id': 2, 'weight': 2, 'flooded': True})]

        self.util_assert_graph(g, expected_graph, expected_nodes, expected_edges)

    def test_graph_edge_sparse(self) -> None:
        file_path: Path = self.graphs_dir / "graph_edge_sparse.txt"
        g: Graph = HurricaneGraphParser(file_path).get_hurricane_graph()

        expected_graph = {'undirected': True, 'equip_time': 1, 'unequip_time': 1, 'kit_slower': 2}
        expected_nodes = [(1, {'has_kit': True, 'num_people': 0}), (2, {'has_kit': False, 'num_people': 1}),
                          (3, {'has_kit': False, 'num_people': 2}), (4, {'has_kit': False, 'num_people': 0}),
                          (5, {'has_kit': True, 'num_people': 0}), (6, {'has_kit': False, 'num_people': 1}),
                          (7, {'has_kit': False, 'num_people': 0}), (8, {'has_kit': False, 'num_people': 3}),
                          (9, {'has_kit': True, 'num_people': 0}), (10, {'has_kit': False, 'num_people': 0})]
        expected_edges = [(1, 2, {'edge_id': 1, 'weight': 1, 'flooded': False}),
                          (2, 3, {'edge_id': 2, 'weight': 2, 'flooded': False}),
                          (3, 4, {'edge_id': 3, 'weight': 3, 'flooded': False}),
                          (5, 6, {'edge_id': 4, 'weight': 1, 'flooded': False}),
                          (6, 7, {'edge_id': 5, 'weight': 2, 'flooded': False}),
                          (8, 9, {'edge_id': 6, 'weight': 1, 'flooded': False})]

        self.util_assert_graph(g, expected_graph, expected_nodes, expected_edges)

    def test_graph_empty(self) -> None:
        file_path: Path = self.graphs_dir / "graph_empty.txt"
        g: Graph = HurricaneGraphParser(file_path).get_hurricane_graph()

        expected_graph = {'undirected': True, 'equip_time': 1, 'unequip_time': 1, 'kit_slower': 1}
        expected_nodes = []
        expected_edges = []

        self.util_assert_graph(g, expected_graph, expected_nodes, expected_edges)

    def test_graph_large(self) -> None:
        file_path: Path = self.graphs_dir / "graph_large.txt"
        g: Graph = HurricaneGraphParser(file_path).get_hurricane_graph()

        expected_graph = {'undirected': True, 'equip_time': 1, 'unequip_time': 2, 'kit_slower': 3}
        expected_nodes = [(1, {'has_kit': False, 'num_people': 1}), (2, {'has_kit': True, 'num_people': 2}),
                          (3, {'has_kit': False, 'num_people': 0}), (4, {'has_kit': True, 'num_people': 0}),
                          (5, {'has_kit': False, 'num_people': 1}), (6, {'has_kit': False, 'num_people': 0})]
        expected_edges = [(1, 2, {'edge_id': 1, 'weight': 2, 'flooded': True}),
                          (1, 3, {'edge_id': 2, 'weight': 3, 'flooded': False}),
                          (2, 4, {'edge_id': 3, 'weight': 1, 'flooded': False}),
                          (2, 6, {'edge_id': 7, 'weight': 5, 'flooded': False}),
                          (3, 4, {'edge_id': 4, 'weight': 2, 'flooded': True}),
                          (4, 5, {'edge_id': 5, 'weight': 1, 'flooded': False}),
                          (5, 6, {'edge_id': 6, 'weight': 3, 'flooded': False})]

        self.util_assert_graph(g, expected_graph, expected_nodes, expected_edges)

    def test_graph_no_attr_nodes_no_edges(self) -> None:
        file_path: Path = self.graphs_dir / "graph_no_attr_nodes_no_edges.txt"
        g: Graph = HurricaneGraphParser(file_path).get_hurricane_graph()

        expected_graph = {'undirected': True, 'equip_time': 1, 'unequip_time': 1, 'kit_slower': 2}
        expected_nodes = [(1, {'has_kit': False, 'num_people': 0}), (2, {'has_kit': False, 'num_people': 0}),
                          (3, {'has_kit': False, 'num_people': 0})]
        expected_edges = []

        self.util_assert_graph(g, expected_graph, expected_nodes, expected_edges)

    def test_graph_no_edges(self) -> None:
        file_path: Path = self.graphs_dir / "graph_no_edges.txt"
        g: Graph = HurricaneGraphParser(file_path).get_hurricane_graph()

        expected_graph = {'undirected': True, 'equip_time': 1, 'unequip_time': 1, 'kit_slower': 2}
        expected_nodes = [(1, {'has_kit': True, 'num_people': 10}), (2, {'has_kit': False, 'num_people': 15}),
                          (3, {'has_kit': True, 'num_people': 20})]
        expected_edges = []

        self.util_assert_graph(g, expected_graph, expected_nodes, expected_edges)
