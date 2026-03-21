import unittest
from pathlib import Path

import networkx as nx

from SearchAgents.utils.hurricane_graph_parser import HurricaneGraphParser
from SearchAgents.agents.a_star_agent import AStarAgent
from SearchAgents.heuristics.hurricane_evacuation_heuristic import HurricaneEvacuationHeuristic


class TestAStarAgent(unittest.TestCase):
    def setUp(self):
        # Using the same graph file as the greedy agent tests
        self.graphs_dir = Path(__file__).parent / "graphs"
        self.file_path: Path = self.graphs_dir / "graph_assignment_example.txt"

        self.world: nx.Graph = HurricaneGraphParser(self.file_path).get_hurricane_graph()
        self.hurricane_heuristic: HurricaneEvacuationHeuristic = HurricaneEvacuationHeuristic(self.world)

    def test_start_node_1_optimal_path(self):
        """
        Start at 1. People at 2 and 4.
        Route 1: 1->2 (flooded) -> ... requires Kit (Time penalty).
        Route 2: 1->3->2->3->4. Cost: 4+1+1+1 = 7.
        Route 2 is faster. The agent should choose to go around.
        """
        agent = AStarAgent(
            self.world,
            initial_node=1,
            heuristic_strategy=self.hurricane_heuristic,
            expansion_limit=10000,
            time_per_expansion=0
        )
        self.assertTrue(agent.get_is_running())

        # Expected Path: 1 -> 3 -> 2 -> 3 -> 4

        # 1. Traverse 1 -> 3
        decision = agent.decide()
        self.assertEqual("traverse 3", decision)
        agent.traverse(3)

        # 2. Traverse 3 -> 2 (Pick up Node 2)
        decision = agent.decide()
        self.assertEqual("traverse 2", decision)
        agent.traverse(2)

        # 3. Traverse 2 -> 3 (Backtrack)
        decision = agent.decide()
        self.assertEqual("traverse 3", decision)
        agent.traverse(3)

        # 4. Traverse 3 -> 4 (Pick up Node 4)
        decision = agent.decide()
        self.assertEqual("traverse 4", decision)
        agent.traverse(4)

        # 5. Terminate
        decision = agent.decide()
        self.assertEqual("terminate", decision)
        agent.terminate()

        self.assertFalse(agent.get_is_running())

    def test_start_node_2_pickup_behavior(self):
        """
        Start at 2. People at 2 and 4.
        The simulator typically clears the starting node's people before the agent acts.
        We must simulate this by clearing node 2 manually, otherwise the agent might
        loop trying to 'arrive' at 2 to pick them up.
        """
        agent = AStarAgent(
            self.world,
            initial_node=2,
            heuristic_strategy=self.hurricane_heuristic,
            expansion_limit=10000,
            time_per_expansion=0
        )

        # Remaining target: Node 4.
        # Path: 2 -> 3 -> 4 (Cost 1+1=2) vs 2->4 (Cost 5)
        # Optimal: 2 -> 3 -> 4

        decision = agent.decide()
        self.assertEqual("traverse 3", decision)
        agent.traverse(3)

        decision = agent.decide()
        self.assertEqual("traverse 4", decision)
        agent.traverse(4)

        decision = agent.decide()
        self.assertEqual("terminate", decision)
        agent.terminate()

    def test_start_node_3_tie_breaking(self):
        """
        Start at 3. People at 2 and 4.
        Neighbors of 3 are 2 (dist 1) and 4 (dist 1).
        Both paths (3->2->3->4 and 3->4->3->2) have equal cost (3).

        Tie-breaking usually favors the node added to the frontier first or with lower ID.
        Assuming neighbors are iterated in order (1, 2, 4), and standard heap stability:
        It should visit 2 first.
        """
        agent = AStarAgent(
            self.world,
            initial_node=3,
            heuristic_strategy=self.hurricane_heuristic,
            expansion_limit=10000,
            time_per_expansion=0
        )

        # 1. 3 -> 2
        decision = agent.decide()
        self.assertEqual("traverse 2", decision)
        agent.traverse(2)

        # 2. 2 -> 3
        decision = agent.decide()
        self.assertEqual("traverse 3", decision)
        agent.traverse(3)

        # 3. 3 -> 4
        decision = agent.decide()
        self.assertEqual("traverse 4", decision)
        agent.traverse(4)

        decision = agent.decide()
        self.assertEqual("terminate", decision)
        agent.terminate()

    def test_expansion_limit_exceeded(self):
        """
        Test that the agent gives up if the problem is too hard for the limit.
        """
        # Set an impossibly low limit
        agent = AStarAgent(
            self.world,
            initial_node=1,
            heuristic_strategy=self.hurricane_heuristic,
            expansion_limit=0,  # Zero expansions allowed
            time_per_expansion=0
        )

        # Should fail to build a plan and terminate immediately
        decision = agent.decide()
        self.assertEqual("terminate", decision)