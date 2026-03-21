import json
from typing import Any
from enum import IntEnum
import networkx as nx


class FloodingStatus(IntEnum):
    """ enumeration for the status of an uncertain edge """
    UNKNOWN = 0
    FLOODED = 1
    CLEAR = 2


class GraphParser:
    """
    utility class vo parse the problem graph from a JSON file
    validates assignment constraints (guaranteed path with P=1 exists).
    """

    @staticmethod
    def parse_graph(graph_file_path: str) -> tuple[nx.Graph, int, int]:
        """
        parses the JSON input into an undirected graph with specific attributes

        Args:
            graph_file_path (str): path to the JSON input file.

        Returns:
            a tuple containing:
                - g (nx.Graph): the graph with edge data (probability, weights) and global attributes (K, EC, UC, FF)
                - agent_start_node (int): the starting vertex (S)
                - agent_target_node (int): the goal vertex (T)

        Raises:
            ValueError: if the graph violates the assignment's 'legal scenario' rules
        """
        g: nx.Graph = nx.Graph()

        with open(graph_file_path) as graph_file:
            graph_structure: dict[str, Any] = json.loads(graph_file.read().strip())

        # extract global parameters
        agent_start_node = int(graph_structure["S"])
        agent_target_node = int(graph_structure["T"])

        # update graph-level attributes
        g.graph.update(
            K=[int(k) for k in graph_structure.get("K", [])],
            EC=int(graph_structure.get("EC", 0)),
            UC=int(graph_structure.get("UC", 0)),
            FF=float(graph_structure.get("FF", 1.0)),
        )

        # parse edges
        for edge_structure in graph_structure.get("E", []):
            src = edge_structure.get("src_node")
            if src is None:
                raise KeyError(f"Edge definition missing source node: {edge_structure}")

            dst = int(edge_structure.get("dst_node"))
            edge_id = int(edge_structure.get("edge_id"))
            weight = float(edge_structure.get("edge_weight"))
            prob = float(edge_structure.get("probability_be_flooded"))

            g.add_edge(
                int(src),
                dst,
                edge_id=edge_id,
                weight=weight,
                probability_be_flooded=prob,
                is_flooded=int(FloodingStatus.UNKNOWN),  # initial knowledge state
            )

        # Validation: verify existence of a risk-free path
        kits: list[int] = g.graph.get("K", [])

        has_stable_to_target: bool = GraphParser.__validate_graph_stable_path(g, agent_start_node, agent_target_node)
        has_stable_to_any_kit = any(
            GraphParser.__validate_graph_stable_path(g, agent_start_node, k) for k in kits
        )
        start_has_kit = agent_start_node in kits

        if not (start_has_kit or has_stable_to_target or has_stable_to_any_kit):
            raise ValueError("Illegal Scenario: No guaranteed (p=0) path from Start to Target or any Kit.")

        return g, agent_start_node, agent_target_node

    @staticmethod
    def __validate_graph_stable_path(g: nx.Graph, start_node: int, target_node: int) -> bool:
        """ checks connectivity using only edges that *never* flood """
        base: nx.Graph = g.to_undirected()

        if start_node not in base or target_node not in base:
            return False

        # filter edges, keep only those with 0 probability of flooding
        safe_edges = [
            (u, v) for u, v, data in base.edges(data=True)
            if float(data.get("probability_be_flooded", 0.0)) == 0.0
        ]

        # create a tmp subgraph with only safe edges
        stable_graph: nx.Graph = base.edge_subgraph(safe_edges)

        # check path existence
        return stable_graph.has_node(start_node) and \
            stable_graph.has_node(target_node) and \
            nx.has_path(stable_graph, start_node, target_node)
