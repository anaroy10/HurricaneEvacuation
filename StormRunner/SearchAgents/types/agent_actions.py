from enum import Enum


class AgentActions(Enum):
    """ actions that an agent can perform in the simulation """
    TRAVERSE = "traverse"
    EQUIP = "equip"
    UNEQUIP = "unequip"
    NO_OP = "no-op"
    TERMINATE = "terminate"
