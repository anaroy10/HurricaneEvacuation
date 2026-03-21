import unittest

import networkx as nx

from SearchAgents.agents.thief_agent import ThiefAgent
from SearchAgents.utils.utils import DummyAgent


class TestThiefAgent(unittest.TestCase):
    def setUp(self):
        self.g = nx.Graph()
        self.g.graph["kit_slower"] = 2
        self.g.graph["equip_time"] = 1

        self.g.add_node(0, num_people=0, has_kit=False)
        self.g.add_node(1, num_people=0, has_kit=True)
        self.g.add_node(2, num_people=0, has_kit=False)
        self.g.add_node(3, num_people=0, has_kit=True)

        self.g.add_edge(0, 1, weight=1)
        self.g.add_edge(1, 2, weight=2)
        self.g.add_edge(2, 3, weight=1)
        self.g.add_edge(0, 3, weight=10)
        self.g.add_edge(1, 3, weight=1, flooded=True)

    def test_move_towards_nearest_kit(self):
        thief = ThiefAgent(self.g, 0)

        decision = thief.decide()
        self.assertEqual(decision, "traverse 1")

        thief.traverse(1)
        self.assertEqual(thief.get_current_node(), 1)
        self.assertEqual(thief.get_total_time(), 1)
        self.assertEqual(thief.get_score(), -1)
        self.assertFalse(thief.get_is_hold_kit())
        self.assertTrue(thief.get_is_running())

        decision = thief.decide()
        self.assertEqual(decision, "equip")

        thief.equip_kit()
        self.assertEqual(thief.get_current_node(), 1)
        self.assertEqual(thief.get_total_time(), 2)
        self.assertEqual(thief.get_score(), -2)
        self.assertTrue(thief.get_is_hold_kit())
        self.assertTrue(thief.get_is_running())

        decision = thief.decide()
        self.assertEqual(decision, "terminate")

        thief.terminate()
        self.assertEqual(thief.get_current_node(), 1)
        self.assertEqual(thief.get_total_time(), 2)
        self.assertEqual(thief.get_score(), -2)
        self.assertTrue(thief.get_is_hold_kit())
        self.assertFalse(thief.get_is_running())


    def test_run_away_from_other_agents(self):
        thief = ThiefAgent(self.g, initial_node=2)
        thief._is_hold_kit = True

        # simulate another agent at node 0
        other = DummyAgent(self.g, 0)

        decision = thief.decide(other_agents=[other])

        # should move to neighbor that maximizes distance from other agent (neighbors: 1,3)
        self.assertEqual(decision, "no-op")

    def test_no_kits_returns_no_op(self):
        # graph with no kits
        g_empty = nx.Graph()
        g_empty.add_node(0, num_people=0, has_kit=False)
        thief = ThiefAgent(g_empty, 0)

        dummy_agent: DummyAgent = DummyAgent(g_empty, 0)

        decision = thief.decide(other_agents=[dummy_agent])
        self.assertEqual(decision, "terminate")
