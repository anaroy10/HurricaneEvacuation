import networkx as nx
from itertools import cycle
from typing import List

from SearchAgents.agents.agent import Agent
from SearchAgents.agents.human_agent import HumanAgent
from SearchAgents.agents.thief_agent import ThiefAgent
from SearchAgents.types.agent_actions import AgentActions
from SearchAgents.types.human_status_codes import HumanStatusCodes, parse_status_message
from SearchAgents.interaction import CLI


class SimulationEngine:
    """ the simulation environment itself """
    def __init__(self, graph: nx.Graph, agents: List[Agent]):
        """ inits a new game """
        self.__world: nx.Graph = graph
        self.__agents: List[Agent] = agents

    def run(self):
        """ the main busy loop of the game """
        for agent in cycle(self.__agents):
            # check if some agent is still running
            if not any(a.get_is_running() for a in self.__agents):
                print("No agents left running.")
                break

            # skip terminated agents
            if not agent.get_is_running():
                continue

            print(agent, end="\n\n")
            self.__process_agent_turn(agent)
            print()

    def __process_agent_turn(self, agent: Agent):
        """ handles logic based on agent type """
        if isinstance(agent, HumanAgent):
            self.__handle_human_turn(agent)
        else:
            self.__handle_auto_turn(agent)

    def __handle_human_turn(self, agent: HumanAgent):
        action_str: str = CLI.get_human_action_input()

        try:
            action_enum = AgentActions(action_str)
        except ValueError:
            print("Bad action")
            return

        code: HumanStatusCodes = HumanStatusCodes.ACTION_NOT_EXISTS

        if action_enum == AgentActions.TRAVERSE:
            dest: int = CLI.get_human_destination()
            code = agent.decide(action=action_str, destination_node=dest)
            if code == HumanStatusCodes.ACTION_POSSIBLE:
                agent.traverse(dest)

        elif action_enum == AgentActions.EQUIP:
            code = agent.decide(action=action_str)
            if code == HumanStatusCodes.ACTION_POSSIBLE:
                agent.equip_kit()

        elif action_enum == AgentActions.UNEQUIP:
            code = agent.decide(action=action_str)
            if code == HumanStatusCodes.ACTION_POSSIBLE:
                agent.unequip_kit()

        elif action_enum == AgentActions.NO_OP:
            code = agent.decide(action=action_str)
            if code == HumanStatusCodes.ACTION_POSSIBLE:
                agent.no_op()

        elif action_enum == AgentActions.TERMINATE:
            agent.terminate()
            return

        if code != HumanStatusCodes.ACTION_POSSIBLE:
            print(parse_status_message(code))

    def __handle_auto_turn(self, agent: Agent):
        # Thief agent requires context of other agents
        if isinstance(agent, ThiefAgent):
            decision = agent.decide(other_agents=[a for a in self.__agents if a != agent])
        else:
            decision = agent.decide()

        tokens: List[str] = decision.split()
        action_enum: AgentActions = AgentActions(tokens[0])

        if action_enum == AgentActions.TRAVERSE:
            agent.traverse(int(tokens[1]))
        elif action_enum == AgentActions.EQUIP:
            agent.equip_kit()
        elif action_enum == AgentActions.UNEQUIP:
            agent.unequip_kit()
        elif action_enum == AgentActions.NO_OP:
            agent.no_op()
        elif action_enum == AgentActions.TERMINATE:
            print(agent)  # print final state before dying
            agent.terminate()