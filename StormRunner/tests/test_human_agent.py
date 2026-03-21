import unittest
import networkx as nx
from SearchAgents.agents.human_agent import HumanAgent
from SearchAgents.types.human_status_codes import HumanStatusCodes, parse_status_message
from SearchAgents.types.agent_actions import AgentActions


class TestHumanAgent(unittest.TestCase):
    def setUp(self):
        self.g = nx.Graph()

        self.g.add_node(0, num_people=2, has_kit=False)
        self.g.add_node(1, num_people=3, has_kit=True)
        self.g.add_node(2, num_people=0, has_kit=False)

        self.g.add_edge(0, 1, weight=1, flooded=False)
        self.g.add_edge(1, 2, weight=2, flooded=True)
        self.g.add_edge(0, 2, weight=5, flooded=False)

    def test_initial_traverse_on_people(self):
        agent = HumanAgent(self.g, initial_node=0)

        self.assertEqual(agent.get_current_node(), 0)
        self.assertEqual(agent.get_people_saved(), 2)
        self.assertEqual(agent.get_total_time(), 0)
        self.assertEqual(agent.get_score(), 2000)
        self.assertTrue(agent.get_is_running())
        self.assertFalse(agent.get_is_hold_kit())

    def test_traverse_action_possible(self):
        agent = HumanAgent(self.g, initial_node=0)
        status = agent.decide(action=AgentActions.TRAVERSE, destination_node=1)
        self.assertEqual(status, HumanStatusCodes.ACTION_POSSIBLE)

        status = agent.decide(action=AgentActions.TRAVERSE, destination_node=2)
        self.assertEqual(status, HumanStatusCodes.ACTION_POSSIBLE)

    def test_traverse_no_available_path(self):
        agent = HumanAgent(self.g, initial_node=0)
        status = agent.decide(action=AgentActions.TRAVERSE, destination_node=99)
        self.assertEqual(status, HumanStatusCodes.NO_AVAILABLE_PATH)

    def test_equip_action(self):
        agent = HumanAgent(self.g, initial_node=1)

        status = agent.decide(action=AgentActions.EQUIP)
        self.assertEqual(status, HumanStatusCodes.ACTION_POSSIBLE)

        self.g.nodes[1]["has_kit"] = False
        status = agent.decide(action=AgentActions.EQUIP)
        self.assertEqual(status, HumanStatusCodes.NO_AMPHIBIAN_KIT)

    def test_unequip_action(self):
        agent = HumanAgent(self.g, initial_node=1)
        # not holding kit
        status = agent.decide(action=AgentActions.UNEQUIP)
        self.assertEqual(status, HumanStatusCodes.UNEQUIP_NOT_HELD)
        # hold kit
        agent._is_hold_kit = True
        # kit already present
        status = agent.decide(action=AgentActions.UNEQUIP)
        self.assertEqual(status, HumanStatusCodes.UNEQUIP_ALREADY_PRESENT)
        # remove kit from node
        self.g.nodes[1]["has_kit"] = False
        status = agent.decide(action=AgentActions.UNEQUIP)
        self.assertEqual(status, HumanStatusCodes.ACTION_POSSIBLE)

    def test_no_op_action(self):
        agent = HumanAgent(self.g, initial_node=0)
        status = agent.decide(action=AgentActions.NO_OP)
        self.assertEqual(status, HumanStatusCodes.ACTION_POSSIBLE)

    def test_unknown_action(self):
        agent = HumanAgent(self.g, initial_node=0)
        status = agent.decide(action="something_invalid")
        self.assertEqual(status, HumanStatusCodes.ACTION_NOT_EXISTS)
