import unittest
from pathlib import Path
from typing import List

import networkx as nx

from SearchAgents.utils.hurricane_graph_parser import HurricaneGraphParser
from SearchAgents.agents.thief_aware_a_star_agent import ThiefAwareAStarAgent
from SearchAgents.heuristics.hurricane_evacuation_heuristic import HurricaneEvacuationHeuristic
from SearchAgents.consts import HAS_KIT


class TestThiefAwareAStarAgent(unittest.TestCase):
    def setUp(self):
        self.graphs_dir = Path(__file__).parent / "graphs"
        self.file_path: Path = self.graphs_dir / "graph_assignment_example.txt"

        self.world: nx.Graph = HurricaneGraphParser(self.file_path).get_hurricane_graph()
        self.hurricane_heuristic = HurricaneEvacuationHeuristic(self.world)

    def run_agent_until_termination(self, agent: ThiefAwareAStarAgent, max_steps=50) -> List[str]:
        """ Helper to execute agent loop and return the sequence of decisions """
        actions = []
        steps = 0
        while agent.get_is_running() and steps < max_steps:
            decision = agent.decide()
            actions.append(decision)

            # Execute state changes on the world so the agent's next observation is correct
            if decision.startswith("traverse"):
                _, target_node = decision.split()
                agent.traverse(int(target_node))
            elif decision == "equip":
                node_data = self.world.nodes[agent._current_node]
                if node_data.get(HAS_KIT):
                    node_data[HAS_KIT] = False
            elif decision == "unequip":
                self.world.nodes[agent._current_node][HAS_KIT] = True
            elif decision == "terminate":
                agent.terminate()

            steps += 1
        return actions

    def test_thief_starts_on_kit_at_4(self):
        """ The test verifies the agent generates a valid successful plan despite the Thief's presence """
        agent = ThiefAwareAStarAgent(
            self.world,
            initial_node=1,
            heuristic_strategy=self.hurricane_heuristic,
            expansion_limit=10000,
            time_per_expansion=0,
            thief_initial_node=4
        )

        actions = self.run_agent_until_termination(agent)

        # assert successful termination
        self.assertEqual(actions[-1], "terminate")
        self.assertFalse(agent.get_is_running())

        # iterate through actions to track location and check equips
        current_node = 1
        for action in actions:
            if action.startswith("traverse"):
                current_node = int(action.split()[1])
            elif action == "equip":
                self.assertNotEqual(current_node, 4,
                                    "Agent tried to equip at Node 4, but Thief should have stolen it!")

    def test_agent_at_3_thief_at_4(self):
        """ Test ensures simply that the agent does not crash and clears the level """
        agent = ThiefAwareAStarAgent(
            self.world,
            initial_node=3,
            heuristic_strategy=self.hurricane_heuristic,
            expansion_limit=10000,
            time_per_expansion=0,
            thief_initial_node=4
        )

        actions = self.run_agent_until_termination(agent)
        self.assertEqual(actions[-1], "terminate")

    def test_no_solution_behavior(self):
        """
        Test extreme constraint (limit=0) to ensure fail-safe behavior works
        similarly to standard A*.
        """
        agent = ThiefAwareAStarAgent(
            self.world,
            initial_node=1,
            heuristic_strategy=self.hurricane_heuristic,
            expansion_limit=0,
            time_per_expansion=0,
            thief_initial_node=4
        )

        decision = agent.decide()
        self.assertEqual(decision, "terminate")
