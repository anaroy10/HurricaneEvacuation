import heapq
from typing import FrozenSet, List, Optional, override

import networkx as nx

from SearchAgents.agents.heuristic_agent import HeuristicAgent
from SearchAgents.types.world_state import WorldState
from SearchAgents.consts import WEIGHT, FLOODED, KIT_SLOWER, HAS_KIT, EQUIP_TIME, UNEQUIP_TIME
from SearchAgents.utils.utils import get_target_nodes


class RealTimeAStarAgent(HeuristicAgent):
    """
    Real-Time A* Agent
    Performs a limited number of expansions (L) before deciding on a move.

    Logic:
    1. Runs A* from the current state with a limit of L expansions.
    2. If the goal is found within L expansions, it caches the full plan and executes it (standard A* behavior).
    3. If the limit is reached without finding the goal:
       - It inspects the frontier (open set).
       - Picks the state with the best heuristic value (lowest f-score).
       - Traces back to find the immediate next action towards that state.
       - Executes that single action and discards the rest of the search tree (re-plans next turn).
    """

    def __init__(self, g: nx.Graph, initial_node: int, heuristic_strategy, expansion_limit: int = 10,
                 time_per_expansion: float = 0):
        super().__init__(g, initial_node, heuristic_strategy, expansion_limit, time_per_expansion)
        self.__cached_plan: List[str] = []

    def __get_remaining_people(self) -> FrozenSet[int]:
        """ Returns a frozenset of node IDs where people are located """
        return frozenset(get_target_nodes(self._world))

    def __get_initial_kits(self) -> FrozenSet[int]:
        """ Returns a frozenset of node IDs where kits are currently located """
        return frozenset(get_target_nodes(self._world, HAS_KIT, False))

    def __run_limited_search(self) -> Optional[str]:
        """
        Runs A* with self._expansion_limit.
        Updates self._cached_plan if goal found.
        Returns the single next action if goal NOT found but valid move exists.
        Returns None if no move possible or global termination.
        """
        start_node = self._current_node
        remaining_people = self.__get_remaining_people()
        # dynamically checking kit locations as they might have moved if we are re-planning
        available_kits = self.__get_initial_kits()

        start_has_kit = self._is_hold_kit

        # Trivial case: goal already satisfied
        if not remaining_people:
            return "terminate"

        # Constants
        equip_time = self._world.graph.get(EQUIP_TIME, 0)
        unequip_time = self._world.graph.get(UNEQUIP_TIME, 0)
        kit_slower = self._world.graph.get(KIT_SLOWER, 1)

        start_state = (start_node, remaining_people, start_has_kit, available_kits)

        ws_start = WorldState(start_node, remaining_people)
        h0 = self._heuristic_strategy.estimate(ws_start)

        counter = 0
        # Heap item: (f, g, count, state, first_action, full_path)
        frontier = []
        heapq.heappush(frontier, (h0, 0, counter, start_state, None, []))

        visited = {}  # state -> g_score
        expansions = 0

        best_f_seen = float('inf')

        while frontier:
            f, g, _, current_state, first_action, path = heapq.heappop(frontier)
            curr_node, curr_people, curr_has_kit, curr_kits = current_state

            # Pruning
            if current_state in visited and visited[current_state] <= g:
                continue
            visited[current_state] = g

            # Track the best heuristic node seen in case we time out
            # We want the node with min f (already sorted by heap, but valid only if not visited worse)
            if f < best_f_seen and first_action is not None:
                best_f_seen = f

            # Check Limits
            if expansions >= self._expansion_limit:
                # Limit reached!
                # If we found a valid direction (best_action_seen), return it.
                # If we haven't expanded children of start yet, we might have nothing.
                if first_action:
                    return first_action

                # If we are at root (first_action is None), we must expand at least once to move.
                # If limit is 0, we do nothing?
                if expansions == 0:
                    pass  # force at least one expansion or return failure?
                else:
                    return first_action  # This is the move towards the best node in the frontier

            expansions += 1

            self._total_time += self._time_per_expansion
            self._score -= self._time_per_expansion

            # Goal Check
            if not curr_people:
                # Goal found within limit! Cache and execute.
                self.__cached_plan = path
                return self.__cached_plan.pop(0)

            # Expand

            # Helper to add to frontier
            def add_successor(next_node, next_people, next_has_kit, next_kits, cost, act_str):
                nonlocal counter
                new_g = g + cost
                new_state_t = (next_node, next_people, next_has_kit, next_kits)

                # Determine first action
                new_first_action = first_action if first_action is not None else act_str
                new_path = path + [act_str]

                if new_state_t not in visited or new_g < visited[new_state_t]:
                    ws_next = WorldState(next_node, next_people)
                    h_val = self._heuristic_strategy.estimate(ws_next)
                    counter += 1
                    heapq.heappush(frontier, (new_g + h_val, new_g, counter, new_state_t, new_first_action, new_path))

            # 1. Traverse
            for nbr in sorted(self._world.neighbors(curr_node)):
                edge_data = self._world.get_edge_data(curr_node, nbr)
                is_flooded = edge_data.get(FLOODED, False)
                w = edge_data.get(WEIGHT, 1)

                if is_flooded and not curr_has_kit:
                    continue

                cost = w * kit_slower if curr_has_kit else w
                n_people = curr_people - {nbr} if nbr in curr_people else curr_people
                add_successor(nbr, n_people, curr_has_kit, curr_kits, cost, f"traverse {nbr}")

            # 2. Equip
            if not curr_has_kit and curr_node in curr_kits:
                n_kits = curr_kits - {curr_node}
                add_successor(curr_node, curr_people, True, n_kits, equip_time, "equip")

            # 3. Unequip
            if curr_has_kit:
                n_kits = curr_kits | {curr_node}
                add_successor(curr_node, curr_people, False, n_kits, unequip_time, "unequip")

        # If frontier empty and no goal -> failed
        return None

    @override
    def decide(self, *args, **kwargs) -> str:
        # 1. Check for cached plan (Goal previously found)
        if self.__cached_plan:
            return self.__cached_plan.pop(0)

        # 2. Check termination condition
        if not self.__get_remaining_people():
            return "terminate"

        # 3. Run RTA*
        action = self.__run_limited_search()

        if action:
            return action

        return "terminate"