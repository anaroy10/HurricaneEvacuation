import heapq
import networkx as nx
from typing import List, Tuple, FrozenSet, Optional, override

from SearchAgents.agents.heuristic_agent import HeuristicAgent
from SearchAgents.heuristics.heuristic_strategy import HeuristicStrategy
from SearchAgents.types.world_state import WorldState
from SearchAgents.consts import WEIGHT, FLOODED, KIT_SLOWER, HAS_KIT, EQUIP_TIME, UNEQUIP_TIME
from SearchAgents.utils.utils import get_target_nodes, remove_flooded_edges


class ThiefAwareAStarAgent(HeuristicAgent):
    """ Bonus 1 Agent: A* Search with Predictive Thief Modeling """
    def __init__(self, g: nx.Graph, initial_node: int, heuristic_strategy: HeuristicStrategy,
                 expansion_limit: int, time_per_expansion: float, thief_initial_node: int):
        super().__init__(g, initial_node, heuristic_strategy, expansion_limit, time_per_expansion)

        self.__plan: List[str] = []
        self.__thief_initial_node = thief_initial_node

        # 1. Create a "Mental Map" for the Thief (Unflooded world)
        self.__unflooded_graph: nx.Graph = remove_flooded_edges(g)

        # 2. Pre-calculate Thief's deterministic path
        self.__thief_trajectory: List[int] = []
        self.__thief_target_kit: Optional[int] = None

        self.__precompute_thief_trajectory()

    def     __precompute_thief_trajectory(self) -> None:
        """
        Simulates the Thief's 'Seek Kit' phase once at startup.
        Stores the sequence of nodes the thief occupies turn-by-turn.
        """
        # Identify all kits currently in the world
        kits: List[int] = get_target_nodes(self._world, HAS_KIT, False)

        if not kits:
            # No kits exist, Thief will terminate
            self.__thief_trajectory = [self.__thief_initial_node]
            self.__thief_target_kit = None
            return

        # Find closest kit (Thief Logic: Min Dist, then Min Node ID)
        best_kit = None
        best_path = []
        best_dist = float("inf")

        for kit in kits:
            try:
                # Thief uses unflooded graph to find kits
                path = nx.shortest_path(self.__unflooded_graph, source=self.__thief_initial_node, target=kit,
                                        weight=WEIGHT)
                dist: int = len(path)

                # Tie-breaking: Distance preferred over Node ID
                if dist < best_dist or (dist == best_dist and kit < (best_kit if best_kit else float("inf"))):
                    best_dist = dist
                    best_kit = kit
                    best_path = path
            except nx.NetworkXNoPath:
                continue

        self.__thief_target_kit = best_kit

        if not self.__thief_target_kit:
            # Trapped
            self.__thief_trajectory = [self.__thief_initial_node]
        else:
            self.__thief_trajectory = best_path

    def __get_next_thief_state(self,
                               current_thief_node: int,
                               is_holding_kit: bool,
                               traj_index: int,
                               available_kits: FrozenSet[int],
                               agent_next_node: int) -> Tuple[int, bool, int]:
        """
        predicts the next decision of the thief agent

        Returns:
            Tuple[int, bool, int]: (new_thief_node, new_is_holding_kit, new_trajectory_index)
        """

        # --- option 1: run away (the Thief already has a kit) ---
        if is_holding_kit:
            candidates = [current_thief_node] + sorted(list(self._world.neighbors(current_thief_node)))
            best_node = current_thief_node
            max_dist = -1

            for cand in candidates:
                try:
                    d = nx.shortest_path_length(self._world, source=cand, target=agent_next_node, weight='weight')
                except nx.NetworkXNoPath:
                    d = float('inf')

                # maximize distance, if needed tie-break: prefer move > stay, then lower ID
                if (d > max_dist or
                        (d == max_dist and best_node != current_thief_node and cand < best_node)):
                    max_dist = d
                    best_node = cand

            return best_node, True, traj_index

        # --- option 2: thief searches for a kit ---

        # validate the cached plan is still valid (thief hasn't stolen the target kit)
        trajectory_valid = (self.__thief_target_kit in available_kits)

        if trajectory_valid and traj_index < len(self.__thief_trajectory):
            # if standing on the kit
            if traj_index == len(self.__thief_trajectory) - 1:
                # Action: equip
                return current_thief_node, True, traj_index  # Next state is holding

            else:
                # Action: traverse
                next_node = self.__thief_trajectory[traj_index + 1]
                return next_node, False, traj_index + 1

        # --- MODE C: FALLBACK (Plan Broken) ---
        else:
            # Agent stole the target kit OR we ran out of path.
            # Recalculate nearest kit dynamically (Expensive but rare).

            if not available_kits:
                return current_thief_node, False, traj_index  # No kits left, stay put

            # Quick search for nearest kit from current location
            best_kit_node = None
            min_len = float('inf')
            best_next_step = current_thief_node

            # Note: In a heavily optimized version, we would cache all-pairs-shortest-paths.
            # Here we do a fresh search because this case implies the 'Golden Path' failed.
            for kit in available_kits:
                try:
                    path = nx.shortest_path(self.__unflooded_graph, source=current_thief_node, target=kit,
                                            weight='weight')
                    if len(path) < min_len or (
                            len(path) == min_len and kit < (best_kit_node if best_kit_node else float('inf'))):
                        min_len = len(path)
                        best_kit_node = kit
                        best_next_step = path[1] if len(path) > 1 else path[0]
                except nx.NetworkXNoPath:
                    continue

            if best_kit_node is None:
                return current_thief_node, False, traj_index

            # If we are already on it
            if best_next_step == current_thief_node:
                return current_thief_node, True, traj_index

            return best_next_step, False, traj_index

    def __build_plan(self) -> None:
        start_node: int = self._current_node
        remaining_people = frozenset(get_target_nodes(self._world))
        available_kits = frozenset(get_target_nodes(self._world, HAS_KIT, False))

        # State:
        # (Agent Node, People Left, Agent Has Kit, Thief Node, Thief Has Kit, Thief Trajectory Index, Available Kits)

        start_state = (
            start_node,
            remaining_people,
            self._is_hold_kit,
            self.__thief_initial_node,
            False,
            0,
            available_kits
        )

        ws_start = WorldState(start_node, remaining_people)
        h0 = self._heuristic_strategy.estimate(ws_start)

        # Heap: (f, g, counter, state, path)
        counter = 0
        frontier = []
        heapq.heappush(frontier, (h0, 0, counter, start_state, []))

        visited = {}
        expansions = 0

        while frontier:
            f, g, _, current_state, path = heapq.heappop(frontier)

            # Unpack State
            (curr_node, curr_people, curr_has_kit,
             t_node, t_has_kit, t_idx, curr_kits) = current_state

            # Visited Check
            if current_state in visited and visited[current_state] <= g:
                continue
            visited[current_state] = g

            # Limits & Goal Check
            if expansions >= self._expansion_limit:
                self.__plan = []
                return
            expansions += 1

            if not curr_people:
                self.__plan = path
                return

            def process_transition(new_agent_node, action_cost, action_name,
                                   is_equip=False, is_unequip=False):
                nonlocal counter

                # 1. Update Agent-side logic
                new_people = curr_people - {new_agent_node} if new_agent_node in curr_people else curr_people

                temp_kits = curr_kits
                new_agent_has_kit = curr_has_kit

                if is_equip:
                    new_agent_has_kit = True
                    temp_kits = temp_kits - {curr_node}
                elif is_unequip:
                    new_agent_has_kit = False
                    temp_kits = temp_kits | {curr_node}  # Drop kit back to world

                # 2. Predict Thief (The Magic Step)
                # We calculate where the thief will be AFTER the agent completes this action.
                new_t_node, new_t_has_kit, new_t_idx = self.__get_next_thief_state(
                    t_node, t_has_kit, t_idx, temp_kits, new_agent_node
                )

                # 3. Handle Kit Stealing Conflict
                # If Thief JUST picked up a kit (transition False -> True), remove it from world
                final_kits = temp_kits
                if not t_has_kit and new_t_has_kit:
                    final_kits = final_kits - {new_t_node}

                # 4. Create New State
                new_state = (new_agent_node, new_people, new_agent_has_kit,
                             new_t_node, new_t_has_kit, new_t_idx, final_kits)

                new_g = g + action_cost
                self._total_time += self._time_per_expansion
                self._score -= self._time_per_expansion

                if new_state not in visited or new_g < visited[new_state]:
                    ws_next = WorldState(new_agent_node, new_people)
                    h_val = self._heuristic_strategy.estimate(ws_next)
                    counter += 1
                    heapq.heappush(frontier, (new_g + h_val, new_g, counter, new_state, path + [action_name]))

            # Action: Traverse
            for nbr in sorted(self._world.neighbors(curr_node)):
                edge_data = self._world.get_edge_data(curr_node, nbr)
                is_flooded = edge_data.get(FLOODED, False)
                base_weight = edge_data.get(WEIGHT, 1)

                if is_flooded and not curr_has_kit:
                    continue

                move_cost = base_weight * self._world.graph.get(KIT_SLOWER, 1) if curr_has_kit else base_weight
                process_transition(nbr, move_cost, f"traverse {nbr}")

            # Action: Equip
            if not curr_has_kit and curr_node in curr_kits:
                process_transition(curr_node, self._world.graph.get(EQUIP_TIME, 0), "equip", is_equip=True)

            # Action: Unequip
            if curr_has_kit:
                process_transition(curr_node, self._world.graph.get(UNEQUIP_TIME, 0), "unequip", is_unequip=True)

        # No solution found
        self.__plan = []

    @override
    def decide(self, *args, **kwargs) -> str:
        if not self.__plan:
            self.__build_plan()

        if not self.__plan:
            # Fallback/Fail behavior
            remaining = get_target_nodes(self._world)
            if remaining:
                return "terminate"
            return "terminate"

        return self.__plan.pop(0)