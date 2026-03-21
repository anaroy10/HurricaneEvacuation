from abc import ABC, abstractmethod
from typing import Any, Union

import networkx as nx

from SearchAgents.types.human_status_codes import HumanStatusCodes
from SearchAgents.consts import EQUIP_TIME,UNEQUIP_TIME, HAS_KIT


class Agent(ABC):
    """ abstract base class that defines the API of a general 'agent' """
    def __init__(self, g: nx.Graph, initial_node: int) -> None:
        """
        inits a new abstract agent

        Args:
            g (nx.Graph): the graph where the agent lives, and simulation occurs
            initial_node (int): initial position (node) on the simulation graph
        """
        self._world: nx.Graph = g
        self._initial_node: int = initial_node

        # the node where the agent is currently standing on
        self._current_node: int = self._initial_node

        # how many people the agent saved, so far
        self._people_saved: int = 0

        # total operational time, so far
        self._total_time: int = 0

        # agent's score := (1000 * people_saved) - total_time
        self._score: int = 0

        # if enabled, the agent can go over flooded edges
        self._is_hold_kit: bool = False

        # agent will run until it terminates (e.g., there is no available path)
        self._is_running: bool = True

    def __str__(self) -> str:
        """ returns string of the agent """
        return (
            f"{self.__class__.__name__}:\n"
            f"  Current node: {self._current_node}\n"
            f"  People saved: {self._people_saved}\n"
            f"  Total time: {self._total_time}\n"
            f"  Score: {self._score}\n"
            f"  Holding kit: {self._is_hold_kit}\n"
            f"  Running: {self._is_running}"
        )

    def get_current_node(self) -> int:
        """ returns the current node index, where the agent stands """
        return self._current_node

    def get_is_hold_kit(self) -> bool:
        """ returns whether the agent is hold an amphibian kit """
        return self._is_hold_kit

    def get_people_saved(self) -> int:
        """ returns the total number of people saved by the agent """
        return self._people_saved

    def get_total_time(self) -> int:
        """ returns the total time the agent took """
        return self._total_time

    def get_score(self) -> int:
        """ returns the score of the agent """
        return self._score

    def get_is_running(self) -> bool:
        """ returns whether the agent still runs or not """
        return self._is_running

    def terminate(self) -> None:
        """ makes the agent stops live in the environment """
        self._is_running = False

    @abstractmethod
    def decide(self, *args: Any, **kwargs: Any) -> Union[HumanStatusCodes, str]:
        """
        runner of the agent, should run all the computations and return the decision to make

        Args:
            args (Any): positional arguments
            kwargs (Any): key word arguments

        Returns:
            action to make by the agent, in the simulation environment
        """
        raise NotImplementedError()

    @abstractmethod
    def traverse(self, dst_node: int, zero_weight: bool = False) -> None:
        """
        traverses the agent around the simulation graph

        Args:
            dst_node (int): new destination node id
            zero_weight (bool): if activated, weight will not be added (used only to traverse self node in startup)
        """
        raise NotImplementedError()

    def equip_kit(self) -> None:
        """ takes an amphibian kit from the current node, recalculates time and score """
        nx.set_node_attributes(self._world, {self._current_node: {HAS_KIT: False}})
        self._is_hold_kit = True
        time_cost: int = self._world.graph.get(EQUIP_TIME, 0)
        self._total_time += time_cost
        self._score -= time_cost

    def unequip_kit(self) -> None:
        """ leaves an amphibian kit at the current node, recalculates time and score """
        nx.set_node_attributes(self._world, {self._current_node: {HAS_KIT: True}})
        self._is_hold_kit = False
        time_cost: int = self._world.graph.get(UNEQUIP_TIME, 0)
        self._total_time += time_cost
        self._score -= time_cost

    def no_op(self) -> None:
        """ skips the current turn, paying time and score penalty fee """
        self._total_time += 1
        self._score -= 1
