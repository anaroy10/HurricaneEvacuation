from typing import override, Any, List
import networkx as nx

from SearchAgents.agents.agent import Agent
from SearchAgents.agents.non_friednly_agent import NonFriendlyAgent
from SearchAgents.consts import HAS_KIT
from SearchAgents.utils.utils import remove_flooded_edges, get_target_nodes


class ThiefAgent(NonFriendlyAgent):
    """ Thief agent, steals amphibian kits then runs away from other agents """
    def __init__(self, g: nx.Graph, initial_node: int) -> None:
        super().__init__(g, initial_node)
        self.__unflooded_world: nx.Graph = remove_flooded_edges(self._world)

    def __move_towards_kit(self) -> str:
        """
        calculates shortest path to the nearest amphibian kit

        Returns:
            decision regarding calculated shortest path
            if no amphibian kits - no-op
            if standing on an amphibian kit - equip
            if we can get amphibian kit - traverse
        """
        # getting all nodes with amphibian kits
        nodes_with_kits: list[int] = get_target_nodes(self._world, HAS_KIT, False)
        if not nodes_with_kits:
            return "terminate"

        # calculating paths to all nodes with kits
        reachable_kits: list[tuple[int, int]] = []
        for node in nodes_with_kits:
            try:
                distance: int = nx.shortest_path_length(
                    self.__unflooded_world, self._current_node, node, weight="weight"
                )
                reachable_kits.append((distance, node))
            except nx.NetworkXNoPath:
                pass
        if not reachable_kits:
            return "no-op"

        # getting the node which is closest and has a kit
        _, target = min(reachable_kits, key=lambda x: (x[0], x[1]))

        # calculating shortest path to the target node (on unflooded graph, because we're yet not holding a kit)
        path: list[int] = nx.shortest_path(self.__unflooded_world, self._current_node, target, weight="weight")

        # if we're standing on the target node
        if len(path) == 1:
            return "equip"

        next_node: int = path[1]
        return f"traverse {next_node}"

    def __run_away(self, other_agents: list[Agent]) -> str:
        # if no other agents, there is no need to run away from anyone
        if not other_agents:
            return "terminate"

        # getting all available nodes from my current position
        # sorted because - "Prefer the lowest-numbered vertices and edge in case of ties"
        candidates: list[int] = [self._current_node] + sorted(self._world.neighbors(self._current_node))

        best_choice_node: int = self._current_node
        best_min_distance: int = -1

        # for each candidate node, compute the distance to the closest agent
        # then choose the node that maximizes this minimum distance
        for n in candidates:
            min_dist_to_any_agent: int = 10**12

            for agent in other_agents:
                try:
                    d: int = nx.shortest_path_length(
                        self._world if self._is_hold_kit else self.__unflooded_world, n, agent._current_node,
                        weight="weight"
                    )
                except nx.NetworkXNoPath:
                    # treat unreachable = extremely far
                    d = 10**12

                if d < min_dist_to_any_agent:
                    min_dist_to_any_agent = d

            # Update the best choice if this node gives a larger minimum distance to any agent.
            # If the distance ties, prefer:
            #   1) Moving over staying on the current node, and
            #   2) The lowest-numbered node among equally-good candidates.
            if (min_dist_to_any_agent > best_min_distance or
                    (min_dist_to_any_agent == best_min_distance and
                     best_choice_node != self._current_node and
                     n < best_choice_node)):
                best_min_distance = min_dist_to_any_agent
                best_choice_node = n

        if best_choice_node == self._current_node:
            return "no-op"

        return f"traverse {best_choice_node}"

    def __clean_agents_list(self, other_agents: List[Agent]) -> List[Agent]:
        """ removes terminated agents """
        for agent in other_agents:
            if not agent.get_is_running():
                other_agents.remove(agent)
        return other_agents

    @override
    def decide(self, **kwargs: Any) -> str:
        other_agents: list[Agent] = self.__clean_agents_list(
            kwargs.get("other_agents", [])
        )

        # If thief does not hold a kit, move to a kit
        if not self._is_hold_kit:
            kit_decision: str = self.__move_towards_kit()

            # kit decision returns no-op only in case there is no path to kit
            # but in that, case we should not skip a turn, but run away
            if kit_decision != "no-op":
                return kit_decision

        # if there is no path to kit or agent already has a kit, run away from other agents
        return self.__run_away(other_agents)
