from collections import deque
from typing import Optional
from dataclasses import dataclass
import networkx as nx

from belief_state_mdp import BeliefState, observe_at_vertex
from graph_parser import FloodingStatus


@dataclass(frozen=True)
class Action:
    """ represents an action the agent can take """
    # "MOVE", "EQUIP", or "UNEQUIP"
    kind: str

    # destination node ID (only for MOVE actions)
    to: Optional[int] = None


def actions_fn(g: nx.Graph, s: BeliefState) -> list[Action]:
    """ returns valid actions available at a concrete belief state """
    acts: list[Action] = []
    kits = set(g.graph.get("K", []))

    # EQUIP action: available if at a kit location and not already equipped
    if (s.v in kits) and (not s.equipped):
        acts.append(Action(kind="EQUIP"))

    # UNEQUIP action: available if equipped
    if s.equipped:
        acts.append(Action(kind="UNEQUIP"))

    # MOVE actions: traverse to neighbors
    for nb in g.neighbors(s.v):
        acts.append(Action(kind="MOVE", to=int(nb)))

    return acts


def transition_fn(
        g: nx.Graph,
        s: BeliefState,
        a: Action,
        edge_id_to_idx: dict[int, int],
        idx_to_edge: list[tuple[int, int, int, float]],
        node_to_incident_idxs: dict[int, list[int]],
) -> tuple[float, list[tuple[float, BeliefState]]]:
    """
    Calculates the transition dynamics and cost

    Returns:
        tuple[float, list[tuple[float, BeliefState]]]: (cost, distribution over next states)
    """
    EC = int(g.graph.get("EC", 0))
    UC = int(g.graph.get("UC", 0))
    FF = float(g.graph.get("FF", 1.0))

    # EQUIP or UNEQUIP
    # deterministic actions that change only the 'equipped' status
    if a.kind == "EQUIP":
        return float(EC), [(1.0, BeliefState(v=s.v, equipped=True, knowledge=s.knowledge))]

    if a.kind == "UNEQUIP":
        return float(UC), [(1.0, BeliefState(v=s.v, equipped=False, knowledge=s.knowledge))]

    # MOVE
    if a.kind != "MOVE" or a.to is None:
        raise ValueError(f"Unknown action: {a}")

    u, v = s.v, a.to
    data = g.get_edge_data(u, v)
    if data is None:
        return float("inf"), []

    w = float(data.get("weight", 1.0))
    edge_id = int(data.get("edge_id"))
    p = float(data.get("probability_be_flooded", 0.0))

    # in case p > 0, we check our knowledge. If p == 0, it's always clear.
    if p == 0.0:
        status = int(FloodingStatus.CLEAR)
    else:
        idx = edge_id_to_idx[edge_id]
        status = s.knowledge[idx]

    # if we know it is FLOODED and we don't have a kit, we cannot traverse
    if (status == int(FloodingStatus.FLOODED)) and (not s.equipped):
        return float("inf"), []

    # calculate move cost
    move_cost = w * (FF if s.equipped else 1.0)

    # determine next state distribution
    dist = observe_at_vertex(
        v=v,
        equipped=s.equipped,
        knowledge=s.knowledge,
        idx_to_edge=idx_to_edge,
        node_to_incident_idxs=node_to_incident_idxs,
    )

    return move_cost, dist


def build_belief_space(
        g: nx.Graph,
        start: int,
        edge_id_to_idx: dict[int, int],
        idx_to_edge: list[tuple[int, int, int, float]],
        node_to_incident_idxs: dict[int, list[int]],
) -> list[BeliefState]:
    """
    performs BFS to generate all reachable belief states from the start configuration
    we only store 'canonical' belief states (states resulting immediately after observation)
    """
    m = len(idx_to_edge)
    initial_knowledge = tuple([int(FloodingStatus.UNKNOWN)] * m)

    # initial state: S node, no kit, unknown edges
    # immediately observe incident edges at S
    init_dist = observe_at_vertex(
        v=start,
        equipped=False,
        knowledge=initial_knowledge,
        idx_to_edge=idx_to_edge,
        node_to_incident_idxs=node_to_incident_idxs,
    )

    q = deque([s for _, s in init_dist])
    visited = set([s for _, s in init_dist])
    reachable_states = list(visited)

    while q:
        s = q.popleft()

        for a in actions_fn(g, s):
            cost, dist = transition_fn(
                g=g, s=s, a=a,
                edge_id_to_idx=edge_id_to_idx,
                idx_to_edge=idx_to_edge,
                node_to_incident_idxs=node_to_incident_idxs,
            )

            # skip invalid moves, a little optimisation
            if not dist or cost == float('inf'):
                continue

            for _, s2 in dist:
                if s2 not in visited:
                    visited.add(s2)
                    reachable_states.append(s2)
                    q.append(s2)

    return reachable_states
