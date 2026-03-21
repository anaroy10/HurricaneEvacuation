from abc import ABC, abstractmethod

import networkx as nx

from SearchAgents.types.world_state import WorldState


class HeuristicStrategy(ABC):
    """ abstract base class for all heuristic evaluations """
    def __init__(self, world_graph: nx.Graph) -> None:
        """
        inits a new abstract heuristic

        Args:
            world_graph (nx.Graph): the simulation graph
        """
        self._world: nx.Graph = world_graph

    @abstractmethod
    def estimate(self, state: WorldState) -> int:
        """
        returns an estimated cost (h-score) from state to goal

        Args:
            state (WorldState): a given state of the world

        Returns:
            int: the calculated h(state) (heuristic score)
        """
        raise NotImplementedError()
