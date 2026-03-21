from enum import Enum


class HumanStatusCodes(Enum):
    """
    human agent differs from another agents by that it is a debug agent, which do not return the next move
    but performs an action that has been decided by the user
    this enum holds status codes to handle outcomes of the human agent
    """
    ACTION_NOT_EXISTS = -1
    ACTION_POSSIBLE = 0
    NO_AVAILABLE_PATH = 1
    NO_AMPHIBIAN_KIT = 2
    UNEQUIP_NOT_HELD = 3
    UNEQUIP_ALREADY_PRESENT = 4


def parse_status_message(status: HumanStatusCodes) -> str:
    """
    convert a HumanStatusCodes enum into a printable message

    Args:
        status (HumanStatusCodes): status code

    Returns:
        message that represents the status code
    """
    messages: dict[HumanStatusCodes, str] = {
        HumanStatusCodes.ACTION_NOT_EXISTS: "Action does not exist",
        HumanStatusCodes.ACTION_POSSIBLE: "Action can be performed",
        HumanStatusCodes.NO_AVAILABLE_PATH: "No available path to destination",
        HumanStatusCodes.NO_AMPHIBIAN_KIT: "Amphibian kit is required for this action",
        HumanStatusCodes.UNEQUIP_NOT_HELD: "Cannot unequip because kit is not held",
        HumanStatusCodes.UNEQUIP_ALREADY_PRESENT: "Kit is already at this node"
    }
    return messages.get(status, "Unknown status code")
