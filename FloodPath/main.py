import sys
import math
from belief_space_mdp import build_belief_space
from belief_state_mdp import build_uncertain_edge_index
from graph_parser import GraphParser
from mdp_solver import (
    format_action,
    format_belief_state,
    precompute_transitions,
    sample_flooding_instance,
    simulate_one_run,
    value_iteration,
)


def print_policy_and_values(states, V, pi, idx_to_edge):
    """ util to print the entire computed policy """
    ordered = sorted(states, key=lambda s: (s.v, int(s.equipped), s.knowledge))
    for s in ordered:
        val = V.get(s, math.inf)
        act = pi.get(s)
        act_str = format_action(act)

        state_str = format_belief_state(s, idx_to_edge)

        if math.isinf(val):
            print(f"{state_str}  V=INF  Action=None (Unreachable)")
        else:
            print(f"{state_str}  V={val:.4f}  Action={act_str}")


def main() -> None:
    argv: list[str] = sys.argv
    argc: int = len(argv)

    if argc != 2:
        print("Usage: python3 main.py <graph_input_json>", file=sys.stderr)
        sys.exit(1)

    input_file_path: str = argv[1]

    # validate graph
    try:
        g, S, T = GraphParser.parse_graph(input_file_path)
    except Exception as e:
        print(f"Error parsing graph: {e}", file=sys.stderr)
        sys.exit(1)

    # build indices for uncertainty (knowledge)
    edge_id_to_idx, idx_to_edge, node_to_incident = build_uncertain_edge_index(g)

    # construct Belief State Space (Reachability Analysis)
    states = build_belief_space(g, S, edge_id_to_idx, idx_to_edge, node_to_incident)
    print(f"Reachable Belief States: {len(states)}")

    # precompute Transitions
    actions_map, trans_map = precompute_transitions(g, states, edge_id_to_idx, idx_to_edge, node_to_incident)

    # run Value Iteration algorithm
    V, pi = value_iteration(states, T, actions_map, trans_map)

    print("\n=== Optimal Policy (Partial View) ===")
    print_policy_and_values(states, V, pi, idx_to_edge)

    # 6. Simulation Loop
    while True:
        raw = input("\nRun simulation? Enter number of runs (or Enter to quit): ").strip()
        if not raw:
            break

        try:
            runs = int(raw)
        except ValueError:
            print("Invalid number.")
            continue

        for i in range(1, runs + 1):
            flooded_inst = sample_flooding_instance(g)
            flooded_ids = sorted([eid for eid, f in flooded_inst.items() if f])

            print(f"\n--- Run {i}/{runs} ---")
            print(f"Real Flooded Edges: {flooded_ids}")

            reached, cost, trace = simulate_one_run(
                g, S, T, edge_id_to_idx, idx_to_edge, node_to_incident, pi, flooded_inst
            )

            for line in trace:
                print(" ", line)

            if reached:
                print(f"Result: SUCCESS. Total Cost: {cost:.2f}")
            else:
                print("Result: FAILURE.")


if __name__ == "__main__":
    main()