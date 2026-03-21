import unittest
from pathlib import Path

import networkx as nx

from SearchAgents.utils.hurricane_graph_parser import HurricaneGraphParser
from SearchAgents.agents.rt_a_star_agent import RealTimeAStarAgent
from SearchAgents.heuristics.hurricane_evacuation_heuristic import HurricaneEvacuationHeuristic


class TestRealTimeAStarAgent(unittest.TestCase):
    def setUp(self):
        self.graphs_dir = Path(__file__).parent / "graphs"
        self.file_path: Path = self.graphs_dir / "graph_assignment_example.txt"

        self.world: nx.Graph = HurricaneGraphParser(self.file_path).get_hurricane_graph()
        self.hurricane_heuristic: HurricaneEvacuationHeuristic = HurricaneEvacuationHeuristic(self.world)

    def test_sufficient_limit_behavior(self):
        """
        If the limit (L) is high enough (e.g., 100), RTA* should find the
        GOAL immediately and behave exactly like standard A*.
        It should cache the plan and not re-calculate on subsequent steps.
        """
        agent = RealTimeAStarAgent(
            self.world,
            initial_node=1,
            heuristic_strategy=self.hurricane_heuristic,
            expansion_limit=100,
            time_per_expansion=0
        )

        self.assertEqual("traverse 3", agent.decide())
        agent.traverse(3)

        self.assertEqual("traverse 2", agent.decide())
        agent.traverse(2)

        self.assertEqual("traverse 3", agent.decide())
        agent.traverse(3)

        self.assertEqual("traverse 4", agent.decide())
        agent.traverse(4)

        self.assertEqual("terminate", agent.decide())

    def test_insufficient_limit_replanning(self):
        """ If the limit (L) is very low (e.g., 1), RTA* cannot find the goal in one pass. """
        agent = RealTimeAStarAgent(
            self.world,
            initial_node=3,
            heuristic_strategy=self.hurricane_heuristic,
            expansion_limit=1,  # EXTREMELY LOW LIMIT
            time_per_expansion=0
        )

        # Agent is at 3. Expands neighbors (2, 4). Limit hit.
        # It picks one (likely 2 or 4 based on tie-breaking since weights are equal).
        move1 = agent.decide()
        self.assertIn(move1, ["traverse 2", "traverse 4"])

        target_node = int(move1.split()[1])
        agent.traverse(target_node)

        move2 = agent.decide()
        self.assertTrue(move2.startswith("traverse") or move2 == "terminate")

    def test_start_node_pickup(self):
        """ Test starting at a node that has people (Node 2). """
        agent = RealTimeAStarAgent(
            self.world,
            initial_node=2,
            heuristic_strategy=self.hurricane_heuristic,
            expansion_limit=10,
            time_per_expansion=0
        )

        self.assertEqual("traverse 3", agent.decide())
        agent.traverse(3)

        self.assertEqual("traverse 4", agent.decide())
        agent.traverse(4)

        self.assertEqual("terminate", agent.decide())