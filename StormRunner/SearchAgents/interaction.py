from typing import List, Type

import networkx as nx

from SearchAgents.agents.a_star_agent import AStarAgent
from SearchAgents.agents.agent import Agent
from SearchAgents.agents.game_agent import GameAgent
from SearchAgents.agents.heuristic_greedy_agent import HeuristicGreedyAgent
from SearchAgents.agents.heuristic_agent import HeuristicAgent
from SearchAgents.agents.rt_a_star_agent import RealTimeAStarAgent
from SearchAgents.types.agent_actions import AgentActions
from SearchAgents.config import AVAILABLE_AGENTS, AGENTS_PRETTY_STR
from SearchAgents.heuristics.hurricane_evacuation_heuristic import HurricaneEvacuationHeuristic
from SearchAgents.consts import LIMIT


class CLI:
    """ supports interaction between the user and the simulator configuration """
    @staticmethod
    def get_game_setup(graph: nx.Graph):
        """ Setup specifically for Assignment 2 Game Mode """
        print("\n--- Game Setup (Assignment 2) ---")
        nodes = list(graph.nodes)

        # 1. Game Type
        print("Select Game Type:")
        print("1. Adversarial (Zero-Sum)")
        print("2. Semi-Cooperative")
        print("3. Fully-Cooperative")
        choice = input("Choice (1-3): ").strip()
        type_map = {'1': 'adversarial', '2': 'semi-cooperative', '3': 'fully-cooperative'}
        game_type = type_map.get(choice, 'adversarial')

        # 2. Deadline
        deadline_input = input("Enter Deadline (turns, default 100): ").strip()
        deadline = int(deadline_input) if deadline_input else 100

        # 3. Agent Positions
        p1 = int(input(f"Agent 0 Start Node (available: {nodes}): ").strip())
        p2 = int(input(f"Agent 1 Start Node (available: {nodes}): ").strip())

        # 4. Create Smart Agents
        # Note: depth_limit is hardcoded to 4, or you can ask the user
        a1 = GameAgent(graph, p1, agent_id=0, game_type=game_type, depth_limit=4)
        a2 = GameAgent(graph, p2, agent_id=1, game_type=game_type, depth_limit=4)

        return [a1, a2], game_type, deadline

    @staticmethod
    def get_initial_agent_setup(simulation_graph: nx.Graph) -> List[Agent]:
        """
        inits all agents according to user choice

        Args:
            simulation_graph (nx.Graph): graph where all agents will live

        Returns:
            List[Agent]: list of initialized agents
        """
        nodes_pretty: str = ", ".join(str(node) for node in simulation_graph.nodes)
        num_agents = int(input("Enter amount of agents: ").strip())

        agents: List[Agent] = []
        for i in range(num_agents):
            # Get Type
            while True:
                agent_type_str = input(f"({i + 1}) Enter agent type (available: {AGENTS_PRETTY_STR}): ").strip().lower()
                if agent_type_str in AVAILABLE_AGENTS:
                    break
                print(f"Invalid agent type, should be one of: {AGENTS_PRETTY_STR}")

            # Get Node
            initial_node: int = int(input(f"Enter initial node for the agent (available: {nodes_pretty}): ").strip())

            # Instantiate
            agent_class: Type[Agent] = AVAILABLE_AGENTS[agent_type_str]

            heuristic: HurricaneEvacuationHeuristic = HurricaneEvacuationHeuristic(simulation_graph)

            if issubclass(agent_class, HeuristicAgent):
                time_per_expansion: float = 0.0

                # Only ask for input if it is A* or RealTime A* (NOT Greedy)
                if not issubclass(agent_class, HeuristicGreedyAgent):
                    time_per_expansion = float(input(f"Enter expansion time cost: ").strip())

                if issubclass(agent_class, HeuristicGreedyAgent):
                    agent: HeuristicGreedyAgent = HeuristicGreedyAgent(
                        simulation_graph, initial_node, heuristic, 0, time_per_expansion
                    )
                elif issubclass(agent_class, AStarAgent):
                    agent: AStarAgent = AStarAgent(
                        simulation_graph, initial_node, heuristic, LIMIT, time_per_expansion
                    )
                elif issubclass(agent_class, RealTimeAStarAgent):
                    limit_expansions: int = int(input("Enter LIMIT: ").strip() or 10)

                    agent: RealTimeAStarAgent = RealTimeAStarAgent(
                        simulation_graph, initial_node, heuristic, limit_expansions, time_per_expansion
                    )

            else:
                agent: Agent = agent_class(simulation_graph, initial_node)

            agents.append(agent)
            print()

        return agents

    @staticmethod
    def get_human_action_input() -> str:
        actions_pretty = ", ".join([action.value.title() for action in AgentActions])
        return input(f"Enter your action (available: {actions_pretty}): ").strip()

    @staticmethod
    def get_human_destination() -> int:
        return int(input("Enter destination node: ").strip())

    @staticmethod
    def print_scoreboard(agents: List[Agent]):
        print("\n#--- Score Board ---#\n")
        for agent in agents:
            print(agent)