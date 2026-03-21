import unittest
from pathlib import Path

import networkx as nx

from SearchAgents.utils.hurricane_graph_parser import HurricaneGraphParser
from SearchAgents.agents.heuristic_greedy_agent import HeuristicGreedyAgent
from SearchAgents.heuristics.hurricane_evacuation_heuristic import HurricaneEvacuationHeuristic


class TestHeuristicGreedyAgent(unittest.TestCase):
    def setUp(self):
        self.graphs_dir  = Path(__file__).parent / "graphs"
        self.file_path: Path = self.graphs_dir / "graph_assignment_example.txt"

        self.world: nx.Graph = HurricaneGraphParser(self.file_path).get_hurricane_graph()
        self.hurricane_heuristic: HurricaneEvacuationHeuristic = HurricaneEvacuationHeuristic(self.world)

    def test_start_node_1(self):
        agent: HeuristicGreedyAgent = HeuristicGreedyAgent(self.world, 1, self.hurricane_heuristic, 0, 0)
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual("traverse 3", decision)
        agent.traverse(3)

        decision = agent.decide()
        self.assertEqual("traverse 2", decision)
        agent.traverse(2)

        decision = agent.decide()
        self.assertEqual("traverse 4", decision)
        agent.traverse(4)

        decision = agent.decide()
        self.assertEqual("terminate", decision)
        agent.terminate()

        self.assertFalse(agent.get_is_running())

    def test_start_node_2(self):
        agent: HeuristicGreedyAgent = HeuristicGreedyAgent(self.world, 2, self.hurricane_heuristic, 0, 0)
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual("traverse 4", decision)
        agent.traverse(4)

        decision = agent.decide()
        self.assertEqual("terminate", decision)
        agent.terminate()

        self.assertFalse(agent.get_is_running())

    def test_start_node_3(self):
        agent: HeuristicGreedyAgent = HeuristicGreedyAgent(self.world, 3, self.hurricane_heuristic, 0, 0)
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual("traverse 2", decision)
        agent.traverse(2)

        decision = agent.decide()
        self.assertEqual("traverse 4", decision)
        agent.traverse(4)

        decision = agent.decide()
        self.assertEqual("terminate", decision)
        agent.terminate()

        self.assertFalse(agent.get_is_running())

    def test_start_node_4(self):
        agent: HeuristicGreedyAgent = HeuristicGreedyAgent(self.world, 4, self.hurricane_heuristic, 0, 0)
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual("traverse 2", decision)
        agent.traverse(2)

        decision = agent.decide()
        self.assertEqual("terminate", decision)
        agent.terminate()

        self.assertFalse(agent.get_is_running())
