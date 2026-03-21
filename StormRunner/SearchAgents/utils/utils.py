from typing import override

from networkx import Graph

from SearchAgents.agents.agent import Agent


def remove_flooded_edges(g: Graph) -> Graph:
    """
    used to create a copy of the graph but without flooded edges

    Returns:
        **copy** of the original graph, removed of all flooded edges
    """
    copy_g: Graph = g.copy()

    flooded_edges = [(u, v) for u, v, attrs in copy_g.edges(data=True) if attrs.get("flooded")]
    copy_g.remove_edges_from(flooded_edges)

    return copy_g


def get_target_nodes(g: Graph, key: str = "num_people", default_value: int = 0) -> list[int]:
    """


    Returns:
        **copy** of the original graph, removed of all flooded edges
    """
    return [
        n for n, data in g.nodes(data=True)
        if data.get(key, default_value) > 0
    ]


class DummyAgent(Agent):
    @override
    def decide(self, *args, **kwargs) -> str: return "noop"

    @override
    def traverse(self, dst_node: int, zero_weight: bool = False) -> None: pass
