import sys
from pathlib import Path

from SearchAgents.game_engine import GameEngine
from SearchAgents.utils.hurricane_graph_parser import HurricaneGraphParser
from SearchAgents.interaction import CLI
from SearchAgents.simulation_engine import SimulationEngine


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m SearchAgents <path_to_graph_file>")
        sys.exit(1)

    # 1. Load Graph
    graph_file = Path(sys.argv[1])
    # Note: Assignments usually use the same graph format
    world_graph = HurricaneGraphParser(graph_file).get_hurricane_graph()

    # 2. Select Mode
    print("Select Mode:")
    print("1. Simulation (Assignment 1 - Search Agents)")
    print("2. Game (Assignment 2 - Game Playing)")
    mode = input("Choice: ").strip()

    if mode == '2':
        # --- Run Assignment 2 Logic ---
        agents, game_type, deadline = CLI.get_game_setup(world_graph)
        engine = GameEngine(world_graph, agents, game_type, deadline)
        engine.run()

    else:
        # --- Run Assignment 1 Logic (Original) ---
        agents = CLI.get_initial_agent_setup(world_graph)
        engine = SimulationEngine(world_graph, agents)
        engine.run()
        CLI.print_scoreboard(agents)
