import heapq
from typing import FrozenSet, List, override

import networkx as nx

from SearchAgents.agents.heuristic_agent import HeuristicAgent
from SearchAgents.types.world_state import WorldState
from SearchAgents.consts import WEIGHT, FLOODED, KIT_SLOWER, HAS_KIT, EQUIP_TIME, UNEQUIP_TIME
from SearchAgents.utils.utils import get_target_nodes


class AStarAgent(HeuristicAgent):
    def __init__(self, g: nx.Graph, initial_node: int, heuristic_strategy, expansion_limit: int,
                 time_per_expansion: float):
        super().__init__(g, initial_node, heuristic_strategy, expansion_limit, time_per_expansion)
        self.__plan: List[str] = []

    def __get_remaining_people(self) -> FrozenSet[int]:
        """ Returns a frozenset of node IDs where people are located """
        return frozenset(get_target_nodes(self._world))

    def __get_initial_kits(self) -> FrozenSet[int]:
        """ Returns a frozenset of node IDs where kits are currently located """
        return frozenset(get_target_nodes(self._world, HAS_KIT, False))

    def __build_plan(self) -> None:
        """
        Run A* from current state to a state where all people are collected.
        Fills self.__plan with the optimal sequence of actions.
        """
        start_node: int = self._current_node
        remaining_people = self.__get_remaining_people()
        # We need to track kit locations because the agent can move them
        available_kits = self.__get_initial_kits()

        start_has_kit: bool = self._is_hold_kit

        # Trivial case: goal already satisfied
        if not remaining_people:
            self.__plan = []
            return

        # Constants from graph globals
        equip_time = self._world.graph.get(EQUIP_TIME, 0)
        unequip_time = self._world.graph.get(UNEQUIP_TIME, 0)
        kit_slower = self._world.graph.get(KIT_SLOWER, 1)

        # Priority Queue for A*: (f_score, g_score, tie_breaker_id, state, path_of_actions)
        # State = (current_node, remaining_people_set, has_kit_bool, available_kits_set)

        start_state = (start_node, remaining_people, start_has_kit, available_kits)

        # create initial state
        ws_start = WorldState(start_node, remaining_people)
        h0 = self._heuristic_strategy.estimate(ws_start)

        # Counter for tie-breaking in heap
        counter = 0

        frontier = []
        heapq.heappush(frontier, (h0, 0, counter, start_state, []))

        visited = {}  # Map state -> best_g_score
        expansions = 0

        while frontier:
            f, g, _, current_state, path = heapq.heappop(frontier)
            curr_node, curr_people, curr_has_kit, curr_kits = current_state

            # Check if we found a better path to this state already
            if current_state in visited and visited[current_state] <= g:
                continue
            visited[current_state] = g

            # Check expansion limit
            if expansions >= self._expansion_limit:
                # Failed to find plan within limit
                self.__plan = []
                return

            expansions += 1

            self._total_time += self._time_per_expansion
            self._score -= self._time_per_expansion

            # Goal Check: No more people to save
            if not curr_people:
                self.__plan = path
                return

            # --- Generate Successors ---

            # 1. Traverse Neighbors
            for nbr in sorted(self._world.neighbors(curr_node)):
                edge_data = self._world.get_edge_data(curr_node, nbr)
                is_flooded = edge_data.get(FLOODED, False)
                base_weight = edge_data.get(WEIGHT, 1)

                # Constraint: Can only traverse flooded if kit is equipped
                if is_flooded and not curr_has_kit:
                    continue

                # Calculate cost
                move_cost = base_weight
                if curr_has_kit:
                    move_cost *= kit_slower

                new_g = g + move_cost

                # People are picked up upon arrival
                new_people = curr_people - {nbr} if nbr in curr_people else curr_people

                new_state = (nbr, new_people, curr_has_kit, curr_kits)
                new_action = f"traverse {nbr}"

                if new_state not in visited or new_g < visited[new_state]:
                    # Calculate Heuristic
                    ws_next = WorldState(nbr, new_people)
                    h_val = self._heuristic_strategy.estimate(ws_next)

                    counter += 1
                    heapq.heappush(frontier, (new_g + h_val, new_g, counter, new_state, path + [new_action]))

            # 2. Equip Kit (if at a node with a kit and not holding one)
            if not curr_has_kit and curr_node in curr_kits:
                new_g = g + equip_time
                # Remove kit from available set (it's now on the agent)
                new_kits = curr_kits - {curr_node}
                new_state = (curr_node, curr_people, True, new_kits)
                new_action = "equip"

                if new_state not in visited or new_g < visited[new_state]:
                    ws_next = WorldState(curr_node, curr_people)
                    h_val = self._heuristic_strategy.estimate(ws_next)
                    counter += 1
                    heapq.heappush(frontier, (new_g + h_val, new_g, counter, new_state, path + [new_action]))

            # 3. Unequip Kit (if holding one)
            if curr_has_kit:
                new_g = g + unequip_time
                # Add kit to current node
                new_kits = curr_kits | {curr_node}
                new_state = (curr_node, curr_people, False, new_kits)
                new_action = "unequip"

                if new_state not in visited or new_g < visited[new_state]:
                    ws_next = WorldState(curr_node, curr_people)
                    h_val = self._heuristic_strategy.estimate(ws_next)
                    counter += 1
                    heapq.heappush(frontier, (new_g + h_val, new_g, counter, new_state, path + [new_action]))

        # If queue empty and no goal found
        self.__plan = []

    @override
    def decide(self, *args, **kwargs) -> str:
        """
        Returns the next action from the calculated plan.
        If no plan exists, it attempts to build one.
        """
        # If we don't have a plan, try to build one
        if not self.__plan:
            self.__build_plan()

        # If we still don't have a plan (search failed or no people left)
        if not self.__plan:
            # If people remain but plan is empty, it means search failed (LIMIT exceeded or unreachable)
            remaining = self.__get_remaining_people()
            if remaining:
                return "terminate"

            return "terminate"

        # Return the next step in the plan
        return self.__plan.pop(0)
