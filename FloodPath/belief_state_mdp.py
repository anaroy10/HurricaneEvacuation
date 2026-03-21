from collections import defaultdict
from dataclasses import dataclass
import networkx as nx
from graph_parser import FloodingStatus


@dataclass(frozen=True)
class BeliefState:
    """ represents a unique state in the Belief MDP """
    # current vertex location of the agent
    v: int

    # true if the agent is carrying the amphibian kit
    equipped: bool

    # a tuple representing the known status of every uncertain edge
    #   maps: index -> FloodingStatus
    knowledge: tuple[int, ...]


def build_uncertain_edge_index(g: nx.Graph) -> \
        tuple[dict[int, int], list[tuple[int, int, int, float]], dict[int, list[int]]]:
    """
    pre-processes the graph to identify and index all uncertain edges

    Returns:
        A tuple containing:
            - edge_id_to_idx (dict[int, int]): maps the edge ID to a static index
            - idx_to_edge (list[tuple[int, int, int, float]]): places all edge's data accordingly with the given index
            - node_to_incident_idxs (dict[int, list[int]]): reflex for nodes to uncertain edges, by index
    """
    # collect uncertain edges
    uncertain: list[tuple[int, int, int, float]] = []
    for u, v, data in g.edges(data=True):
        p = float(data.get("probability_be_flooded", 0.0))
        if p > 0.0:
            edge_id = int(data.get("edge_id"))
            uncertain.append((edge_id, int(u), int(v), p))

    # validate assignment constraint
    if len(uncertain) > 10:
        raise ValueError(f"Too many uncertain edges: {len(uncertain)} (max is 10)")

    # sort by edge_id to ensure deterministic ordering of the knowledge tuple
    uncertain.sort(key=lambda t: t[0])

    edge_id_to_idx = {edge_id: i for i, (edge_id, _, _, _) in enumerate(uncertain)}

    # store clean tuples for easy access later: (u, v, edge_id, p)
    idx_to_edge = [
        (u, v, edge_id, p) for (edge_id, u, v, p) in uncertain
    ]

    # map nodes to incident uncertain edge indices
    #   this allows O(1) lookup of which edges are revealed when arriving at a node
    node_to_incident_idxs = defaultdict(list)
    for idx, (u, v, edge_id, p) in enumerate(idx_to_edge):
        node_to_incident_idxs[u].append(idx)
        node_to_incident_idxs[v].append(idx)

    for node in node_to_incident_idxs:
        node_to_incident_idxs[node].sort()

    return edge_id_to_idx, idx_to_edge, dict(node_to_incident_idxs)


def observe_at_vertex(
        v: int,
        equipped: bool,
        knowledge: tuple[int, ...],
        idx_to_edge: list[tuple[int, int, int, float]],
        node_to_incident_idxs: dict[int, list[int]],
) -> list[tuple[float, BeliefState]]:
    """
    simulates the observation step when the agent arrives at a vertex

    any incident edges that are currently UNKNOWN are revealed,
    this creates a probability distribution over new belief states

    Returns:
        list[tuple[float, BeliefState]]: a list of (probability, BeliefState) tuples.
    """
    incident = node_to_incident_idxs.get(v, [])

    # identify which incident edges are currently unknown to the agent
    unknown_idxs = [idx for idx in incident if knowledge[idx] == int(FloodingStatus.UNKNOWN)]

    # if all incident edges are already known, state remains deterministic
    if not unknown_idxs:
        return [(1.0, BeliefState(v=v, equipped=equipped, knowledge=knowledge))]

    # iteratively expand the distribution for each unknown edge
    dist: list[tuple[float, list[int]]] = [(1.0, list(knowledge))]

    for idx in unknown_idxs:
        _, _, _, p = idx_to_edge[idx]
        new_dist: list[tuple[float, list[int]]] = []

        for prob, klist in dist:
            # edge is FLOODED
            k_f = klist.copy()
            k_f[idx] = int(FloodingStatus.FLOODED)
            new_dist.append((prob * p, k_f))

            # edge is CLEAR
            k_c = klist.copy()
            k_c[idx] = int(FloodingStatus.CLEAR)
            new_dist.append((prob * (1.0 - p), k_c))

        dist = new_dist

    # convert mutable lists back to immutable BeliefState objects and merge duplicates
    merged: dict[BeliefState, float] = defaultdict(float)
    for prob, klist in dist:
        s2 = BeliefState(v=v, equipped=equipped, knowledge=tuple(klist))
        merged[s2] += prob

    return [(p, s) for s, p in merged.items()]
