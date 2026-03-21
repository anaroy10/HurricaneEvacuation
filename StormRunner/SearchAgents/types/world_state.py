from typing import FrozenSet
from dataclasses import dataclass

@dataclass(frozen=True)
class WorldState:
    """ represents an agent's state of the environment """
    # current node where agent stands on
    agent_location: int

    # remaining nodes with people
    people_locations: FrozenSet[int]
