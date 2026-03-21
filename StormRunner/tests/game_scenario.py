import networkx as nx
from SearchAgents.agents.game_agent import GameAgent
from SearchAgents.types.game_state import GameState
from SearchAgents.consts import NUM_PEOPLE, WEIGHT


def print_separator(char='-', length=60):
    print(char * length)


def run_full_simulation(title, graph, a0_start, a1_start, description):
    print(f"\n{'=' * 30}\n{title}\n{'=' * 30}")
    print(f"Scenario: {description}\n")

    modes = ['adversarial', 'semi-cooperative', 'fully-cooperative']

    for mode in modes:
        print(f"\n>>> Simulation Mode: {mode.upper()}")
        print_separator()

        # 1. Setup Initial State
        agents_data = [
            {'pos': a0_start, 'score': 0, 'has_kit': False},
            {'pos': a1_start, 'score': 0, 'has_kit': False}
        ]
        # Clock 0, Deadline 6 turns (Short enough for demo)
        state = GameState(graph, agents_data, 0, 6)

        # Create Agents
        a0 = GameAgent(graph, a0_start, 0, mode, depth_limit=4)
        a1 = GameAgent(graph, a1_start, 1, mode, depth_limit=4)
        agents = [a0, a1]

        # 2. Run Simulation Loop
        game_over = False
        turn = 0

        # Header
        print(f"{'Turn':<5} | {'Agent':<5} | {'Action':<15} | {'Scores (A0 - A1)'}")
        print_separator()

        while not game_over and turn < 6:
            # Each "Turn" involves moves from Agent 0 AND Agent 1
            for i in range(2):
                if state.is_game_over():
                    game_over = True
                    break

                # Decision
                action = agents[i].decide(state)

                # Apply Action (Update State)
                # Note: generate_successor handles cost, movement, and score updates
                state = state.generate_successor(i, action)

                # Print Step
                scores = f"{state.get_score(0)} - {state.get_score(1)}"
                print(f"{state.turn_count:<5} | {f'A{i}':<5} | {action:<15} | {scores}")

            turn += 1

        # Final Result Summary
        s0 = state.get_score(0)
        s1 = state.get_score(1)
        print_separator()
        print(f"FINAL RESULT ({mode}): A0={s0}, A1={s1} | Diff={s0 - s1} | Sum={s0 + s1}")


def run_altruism_test():
    """
    A0 can take 100 (Greedy) or 10 (Altruist).
    - If A0 takes 100 (Node 2), A1 gets NOTHING.
    - If A0 takes 10 (Node 1), A1 can reach Node 2 (100).
    """
    g = nx.Graph()
    # Node 0: A0 Start
    # Node 1: Small Prize (10)
    # Node 2: Big Prize (100)
    # Node 3: A1 Start (Connected ONLY to 2)
    g.add_node(0, **{NUM_PEOPLE: 0})
    g.add_node(1, **{NUM_PEOPLE: 10})
    g.add_node(2, **{NUM_PEOPLE: 100})
    g.add_node(3, **{NUM_PEOPLE: 0})

    g.add_edge(0, 1, weight=1)
    g.add_edge(0, 2, weight=1)
    g.add_edge(3, 2, weight=1)

    desc = "A0 chooses: Take 100 (A1 gets 0) OR Take 10 (A1 gets 100)."
    run_full_simulation("ALTRUISM TEST (Long Run)", g, 0, 3, desc)


def run_spite_test():
    """
    A0 can take 105 or 100.
    - Node 1 (100 pts): Blocks A1. A1 gets 0. (Diff +100)
    - Node 2 (105 pts): Doesn't block A1. A1 takes Node 1 (100). (Diff +5)
    """
    g = nx.Graph()
    g.add_node(0, **{NUM_PEOPLE: 0})  # A0 Start
    g.add_node(1, **{NUM_PEOPLE: 100})  # Spite Target
    g.add_node(2, **{NUM_PEOPLE: 105})  # Greedy Target
    g.add_node(3, **{NUM_PEOPLE: 0})  # A1 Start

    # A0 connections
    g.add_edge(0, 1, weight=1)
    g.add_edge(0, 2, weight=1)

    # A1 can ONLY reach Node 1
    g.add_edge(3, 1, weight=1)

    desc = "A0 chooses: Take 105 (A1 gets 100) OR Take 100 (A1 gets 0)."
    run_full_simulation("SPITE TEST (Long Run)", g, 0, 3, desc)


if __name__ == "__main__":
    run_altruism_test()
    run_spite_test()