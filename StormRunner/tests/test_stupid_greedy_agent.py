import unittest

import networkx as nx

from SearchAgents.agents.stupid_greedy_agent import StupidGreedyAgent


class TestStupidGreedyAgent(unittest.TestCase):
    def setUp(self):
        self.g = nx.Graph()

        self.g.add_node(0, num_people=1)
        self.g.add_node(1, num_people=2)
        self.g.add_node(2, num_people=0)
        self.g.add_node(3, num_people=3)

        self.g.add_edge(0, 1, weight=1, flooded=False)
        self.g.add_edge(1, 2, weight=1, flooded=False)
        self.g.add_edge(2, 3, weight=1, flooded=False)
        self.g.add_edge(0, 3, weight=10, flooded=False)
        self.g.add_edge(1, 3, weight=1, flooded=True)

    def test_decide_traverses_to_node_with_most_people(self):
        agent = StupidGreedyAgent(self.g, 0)

        decision = agent.decide()
        self.assertEqual(decision, "traverse 1")

        self.assertEqual(agent.get_current_node(), 0)
        self.assertEqual(agent.get_score(), 1000)
        self.assertEqual(agent.get_people_saved(), 1)
        self.assertEqual(agent.get_total_time(), 0)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertTrue(agent.get_is_running())

        agent.traverse(1)
        self.assertEqual(agent.get_current_node(), 1)
        self.assertEqual(agent.get_score(), 2999)
        self.assertEqual(agent.get_people_saved(), 3)
        self.assertEqual(agent.get_total_time(), 1)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual(decision, "traverse 2")

        agent.traverse(2)
        self.assertEqual(agent.get_current_node(), 2)
        self.assertEqual(agent.get_score(), 2998)
        self.assertEqual(agent.get_people_saved(), 3)
        self.assertEqual(agent.get_total_time(), 2)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual(decision, "traverse 3")

        agent.traverse(3)
        self.assertEqual(agent.get_current_node(), 3)
        self.assertEqual(agent.get_score(), 5997)
        self.assertEqual(agent.get_people_saved(), 6)
        self.assertEqual(agent.get_total_time(), 3)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual(decision, "terminate")

        agent.terminate()
        self.assertEqual(agent.get_current_node(), 3)
        self.assertEqual(agent.get_score(), 5997)
        self.assertEqual(agent.get_people_saved(), 6)
        self.assertEqual(agent.get_total_time(), 3)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertFalse(agent.get_is_running())


    def test_decide_avoids_flooded_edges(self):
        agent = StupidGreedyAgent(self.g, 1)

        decision = agent.decide()
        self.assertEqual(decision, "traverse 2")

        self.assertEqual(agent.get_current_node(), 1)
        self.assertEqual(agent.get_score(), 2000)
        self.assertEqual(agent.get_people_saved(), 2)
        self.assertEqual(agent.get_total_time(), 0)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertTrue(agent.get_is_running())

        agent.traverse(2)
        self.assertEqual(agent.get_current_node(), 2)
        self.assertEqual(agent.get_score(), 1999)
        self.assertEqual(agent.get_people_saved(), 2)
        self.assertEqual(agent.get_total_time(), 1)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual(decision, "traverse 3")

        agent.traverse(3)
        self.assertEqual(agent.get_current_node(), 3)
        self.assertEqual(agent.get_score(), 4998)
        self.assertEqual(agent.get_people_saved(), 5)
        self.assertEqual(agent.get_total_time(), 2)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual(decision, "traverse 2")

        agent.traverse(2)
        self.assertEqual(agent.get_current_node(), 2)
        self.assertEqual(agent.get_score(), 4997)
        self.assertEqual(agent.get_people_saved(), 5)
        self.assertEqual(agent.get_total_time(), 3)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual(decision, "traverse 1")

        agent.traverse(1)
        self.assertEqual(agent.get_current_node(), 1)
        self.assertEqual(agent.get_score(), 4996)
        self.assertEqual(agent.get_people_saved(), 5)
        self.assertEqual(agent.get_total_time(), 4)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual(decision, "traverse 0")

        agent.traverse(0)
        self.assertEqual(agent.get_current_node(), 0)
        self.assertEqual(agent.get_score(), 5995)
        self.assertEqual(agent.get_people_saved(), 6)
        self.assertEqual(agent.get_total_time(), 5)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual(decision, "terminate")

        agent.terminate()
        self.assertEqual(agent.get_current_node(), 0)
        self.assertEqual(agent.get_score(), 5995)
        self.assertEqual(agent.get_people_saved(), 6)
        self.assertEqual(agent.get_total_time(), 5)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertFalse(agent.get_is_running())

    def test_decide_terminates_if_no_people(self):
        g_empty = nx.Graph()
        g_empty.add_node(0, num_people=0)
        agent = StupidGreedyAgent(g_empty, 0)

        self.assertEqual(agent.get_current_node(), 0)

        decision = agent.decide()
        self.assertEqual(decision, "terminate")

        agent.terminate()
        self.assertEqual(agent.get_current_node(), 0)
        self.assertEqual(agent.get_score(), 0)
        self.assertEqual(agent.get_people_saved(), 0)
        self.assertEqual(agent.get_total_time(), 0)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertFalse(agent._is_running)

    def test_initial_traverse_if_start_on_people(self):
        g_start = nx.Graph()
        g_start.add_node(0, num_people=5)
        agent = StupidGreedyAgent(g_start, 0)

        self.assertEqual(agent.get_current_node(), 0)
        self.assertEqual(agent.get_score(), 5000)
        self.assertEqual(agent.get_people_saved(), 5)
        self.assertEqual(agent.get_total_time(), 0)
        self.assertFalse(agent.get_is_hold_kit(), False)
        self.assertTrue(agent.get_is_running())

        decision = agent.decide()
        self.assertEqual(decision, "terminate")

        agent.terminate()
        self.assertEqual(agent.get_current_node(), 0)
        self.assertEqual(agent.get_score(), 5000)
        self.assertEqual(agent.get_people_saved(), 5)
        self.assertEqual(agent.get_total_time(), 0)
        self.assertFalse(agent.get_is_hold_kit())
        self.assertFalse(agent.get_is_running())
