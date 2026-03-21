from typing import override, Any

from networkx import Graph

from SearchAgents.agents.friendly_agent import FriendlyAgent
from SearchAgents.consts import HAS_KIT, FLOODED
from SearchAgents.types.agent_actions import AgentActions
from SearchAgents.types.human_status_codes import HumanStatusCodes


class HumanAgent(FriendlyAgent):
    """ a debug agent """
    def __init__(self, g: Graph, initial_node: int) -> None:
        """ inits a new human agent """
        super().__init__(g, initial_node)

    @override
    def decide(self, **kwargs: Any) -> HumanStatusCodes:
        """ debug agent do not return the action to perform, but a status code """
        try:
            action: AgentActions = AgentActions(kwargs.get("action"))
        except ValueError:
            return HumanStatusCodes.ACTION_NOT_EXISTS

        current: int = self._current_node
        node_data: dict = self._world.nodes[current]
        has_kit_here: bool = node_data.get(HAS_KIT, False)

        if action == AgentActions.TRAVERSE:
            destination: int = kwargs.get("destination_node")
            edge_exists = self._world.has_edge(current, destination)
            edge_flooded = edge_exists and self._world[current][destination].get(FLOODED, False)

            if not edge_exists:
                return HumanStatusCodes.NO_AVAILABLE_PATH
            if edge_flooded and not self._is_hold_kit:
                return HumanStatusCodes.NO_AMPHIBIAN_KIT
            return HumanStatusCodes.ACTION_POSSIBLE

        elif action == AgentActions.EQUIP:
            if not self._is_hold_kit and has_kit_here:
                return HumanStatusCodes.ACTION_POSSIBLE
            return HumanStatusCodes.NO_AMPHIBIAN_KIT

        elif action == AgentActions.UNEQUIP:
            if not self._is_hold_kit:
                return HumanStatusCodes.UNEQUIP_NOT_HELD
            if has_kit_here:
                return HumanStatusCodes.UNEQUIP_ALREADY_PRESENT
            return HumanStatusCodes.ACTION_POSSIBLE

        elif action == AgentActions.NO_OP:
            return HumanStatusCodes.ACTION_POSSIBLE

        return HumanStatusCodes.ACTION_NOT_EXISTS
