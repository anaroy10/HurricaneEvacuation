from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional
import math
import random
import networkx as nx

from belief_state_mdp import BeliefState
from belief_space_mdp import Action, actions_fn, transition_fn
from graph_parser import FloodingStatus


def format_action(a: Optional[Action]) -> str:
    """ util to stringify actions for printing """
    if a is None:
        return "<NONE>"
    if a.kind == "MOVE":
        return f"MOVE({a.to})"
    return a.kind


def format_belief_state(s: BeliefState, idx_to_edge: list[tuple[int, int, int, float]]) -> str:
    """ util to pretty-print the belief state knowledge """
    sym = {
        int(FloodingStatus.UNKNOWN): "U",
        int(FloodingStatus.CLEAR): "C",
        int(FloodingStatus.FLOODED): "F",
    }
    parts = []
    for i, (_, _, edge_id, _) in enumerate(idx_to_edge):
        parts.append(f"e{edge_id}={sym.get(s.knowledge[i], '?')}")
    k_str = ",".join(parts)
    return f"v={s.v} eq={int(s.equipped)} [{k_str}]"


@dataclass(frozen=True)
class TransitionCacheEntry:
    """ cache for operations and their cost, before value iteration """
    cost: float
    dist: list[dict[float, BeliefState]]


def precompute_transitions(
        g: nx.Graph,
        states: Iterable[BeliefState],
        edge_id_to_idx: dict[int, int],
        idx_to_edge: list[tuple[int, int, int, float]],
        node_to_incident_idxs: dict[int, list[int]],
) -> tuple[dict[BeliefState, list[Action]], dict[tuple[BeliefState, Action], TransitionCacheEntry]]:
    """ precomputes actions and transitions for all states to speed up Value Iteration """
    actions_map = {}
    trans_map = {}

    for s in states:
        acts = actions_fn(g, s)
        actions_map[s] = acts
        for a in acts:
            cost, dist = transition_fn(
                g, s, a, edge_id_to_idx, idx_to_edge, node_to_incident_idxs
            )
            trans_map[(s, a)] = TransitionCacheEntry(cost=cost, dist=dist)

    return actions_map, trans_map


def value_iteration(
        states: Iterable[BeliefState],
        target_v: int,
        actions_map: dict[BeliefState, list[Action]],
        trans_map: dict[tuple[BeliefState, Action], TransitionCacheEntry],
        eps: float = 1e-9,
        max_iters: int = 50_000,
) -> tuple[dict[BeliefState, float], dict[BeliefState, Optional[Action]]]:
    """
    solves the Stochastic Shortest Path MDP using Value Iteration

    Bellman Equation: V(s) = min_a [ Cost(s,a) + sum_s' P(s'|s,a) * V(s') ]
    Target states are 0, else are +inf
    """
    # initialize values: 0 for target, inf for others
    V: dict[BeliefState, float] = {s: (0.0 if s.v == target_v else math.inf) for s in states}
    pi: dict[BeliefState, Optional[Action]] = {s: None for s in states}

    ordered_states = sorted(states, key=lambda s: (s.v, int(s.equipped), s.knowledge))

    for _ in range(max_iters):
        delta = 0.0

        for s in ordered_states:
            if s.v == target_v:
                continue

            best_val = math.inf
            best_act = None

            # iterate over all possible actions
            for a in actions_map.get(s, []):
                entry = trans_map.get((s, a))
                if not entry or not entry.dist or math.isinf(entry.cost):
                    continue

                # expected value of next state
                exp_next = sum(p * V[s2] for p, s2 in entry.dist)

                # Q-value
                q = entry.cost + exp_next

                if q < best_val:
                    best_val = q
                    best_act = a

            old_val = V[s]
            V[s] = best_val
            pi[s] = best_act

            # check convergence
            if not (math.isinf(old_val) and math.isinf(best_val)):
                delta = max(delta, abs(best_val - old_val))

        if delta < eps:
            break

    return V, pi


# --- Simulation Logic ---

def sample_flooding_instance(g: nx.Graph) -> dict[int, bool]:
    """
    generates a concrete world

    Returns:
        dict[int, bool]: a dict mapping edge_id -> is_flooded
    """
    inst = {}
    for _, _, data in g.edges(data=True):
        p = float(data.get("probability_be_flooded", 0.0))
        if p > 0.0:
            edge_id = int(data["edge_id"])
            inst[edge_id] = (random.random() < p)
    return inst


def observe_deterministic(
        v: int,
        equipped: bool,
        knowledge: tuple[int, ...],
        idx_to_edge: list[tuple[int, int, int, float]],
        node_to_incident_idxs: dict[int, list[int]],
        flooded_instance: dict[int, bool],
) -> BeliefState:
    """
    Updates the belief state based on the *actual* ground truth (flooded_instance)
    rather than branching probabilities. Used during simulation.
    """
    k = list(knowledge)
    # Check all edges incident to current vertex 'v'
    for idx in node_to_incident_idxs.get(v, []):
        # If we already know the status, skip
        if k[idx] != int(FloodingStatus.UNKNOWN):
            continue

        _, _, edge_id, _ = idx_to_edge[idx]

        # Look up ground truth
        is_flooded = flooded_instance.get(edge_id, False)
        k[idx] = int(FloodingStatus.FLOODED if is_flooded else FloodingStatus.CLEAR)

    return BeliefState(v=v, equipped=equipped, knowledge=tuple(k))


def simulate_one_run(
        g: nx.Graph,
        start: int,
        target: int,
        edge_id_to_idx: dict[int, int],
        idx_to_edge: list[tuple[int, int, int, float]],
        node_to_incident_idxs: dict[int, list[int]],
        policy: dict[BeliefState, Optional[Action]],
        flooded_instance: dict[int, bool],
        step_limit: int = 1000,
) -> tuple[bool, float, list[str]]:
    """ simulates the agent navigating the graph using the computed policy against a specific flooding instance """
    m = len(idx_to_edge)
    # Initial state
    s = BeliefState(v=start, equipped=False, knowledge=tuple([int(FloodingStatus.UNKNOWN)] * m))

    # Initial observation at start node
    s = observe_deterministic(
        s.v, s.equipped, s.knowledge, idx_to_edge, node_to_incident_idxs, flooded_instance
    )

    total_cost = 0.0
    trace = [f"START at {s.v}"]

    for _ in range(step_limit):
        if s.v == target:
            trace.append(f"REACHED TARGET {target}")
            return True, total_cost, trace

        a = policy.get(s)
        if a is None:
            trace.append("NO ACTION (Dead End or Unreachable State)")
            return False, math.inf, trace

        # --- Execute Action ---
        if a.kind == "EQUIP":
            cost = float(g.graph.get("EC", 0))
            total_cost += cost
            s = BeliefState(v=s.v, equipped=True, knowledge=s.knowledge)
            trace.append(f"EQUIP (cost {cost})")
            continue

        if a.kind == "UNEQUIP":
            cost = float(g.graph.get("UC", 0))
            total_cost += cost
            s = BeliefState(v=s.v, equipped=False, knowledge=s.knowledge)
            trace.append(f"UNEQUIP (cost {cost})")
            continue

        if a.kind == "MOVE":
            u, v = s.v, int(a.to)
            data = g.get_edge_data(u, v)
            edge_id = int(data["edge_id"])
            w = float(data.get("weight", 1.0))
            ff = float(g.graph.get("FF", 1.0))

            # check if blocked in ground truth
            is_flooded_reality = flooded_instance.get(edge_id, False)

            # if edge is actually flooded, and we have no kit, we are blocked
            if is_flooded_reality and not s.equipped:
                # We tried to move, but failed
                trace.append(f"BLOCKED moving {u}->{v} (Edge {edge_id} flooded)")

                # manually update knowledge for that specific edge if it wasn't already
                k_list = list(s.knowledge)
                if edge_id in edge_id_to_idx:
                    k_list[edge_id_to_idx[edge_id]] = int(FloodingStatus.FLOODED)
                s = BeliefState(v=s.v, equipped=s.equipped, knowledge=tuple(k_list))

                continue

            # Move successful
            move_cost = w * (ff if s.equipped else 1.0)
            total_cost += move_cost
            trace.append(f"MOVE {u}->{v} (cost {move_cost:.2f})")

            # Arrive at v
            s = BeliefState(v=v, equipped=s.equipped, knowledge=s.knowledge)
            # Observe surroundings
            s = observe_deterministic(
                s.v, s.equipped, s.knowledge, idx_to_edge, node_to_incident_idxs, flooded_instance
            )

    trace.append("STEP LIMIT REACHED")
    return False, math.inf, trace
