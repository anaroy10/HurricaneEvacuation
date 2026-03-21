from typing import Dict, Type
from SearchAgents.agents.agent import Agent
from SearchAgents.agents.human_agent import HumanAgent
from SearchAgents.agents.stupid_greedy_agent import StupidGreedyAgent
from SearchAgents.agents.thief_agent import ThiefAgent
from SearchAgents.agents.heuristic_greedy_agent import HeuristicGreedyAgent
from SearchAgents.agents.a_star_agent import AStarAgent
from SearchAgents.agents.rt_a_star_agent import RealTimeAStarAgent


AVAILABLE_AGENTS: Dict[str, Type[Agent]] = {
    "human": HumanAgent,
    "stupid greedy": StupidGreedyAgent,
    "thief": ThiefAgent,
    "heuristic greedy": HeuristicGreedyAgent,
    "optimal": AStarAgent,
    "real time": RealTimeAStarAgent
}

AGENTS_PRETTY_STR: str = ", ".join(agent.title() for agent in AVAILABLE_AGENTS)
