from abc import ABC, abstractmethod
from typing import Callable, List, Any

import networkx as nx

from SearchAgents.agents.friendly_agent import FriendlyAgent
from SearchAgents.heuristics.heuristic_strategy import HeuristicStrategy

class HeuristicAgent(FriendlyAgent):
    """ defines an abstract heuristic agent """
    def __init__(
            self, g: nx.Graph, initial_node: int,
            heuristic_strategy: HeuristicStrategy, expansion_limit: int, time_per_expansion: float
    ) -> None:
        super().__init__(g, initial_node)

        self._heuristic_strategy: HeuristicStrategy = heuristic_strategy
        self._expansion_limit: int = expansion_limit
        self._time_per_expansion: float = time_per_expansion
