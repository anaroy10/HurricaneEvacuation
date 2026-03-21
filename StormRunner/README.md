# StormRunner

Graph-based hurricane evacuation project focused on search, simulation, and multi-agent decision making.

## Overview
This project models hurricane evacuation on a graph-based environment with dynamic conditions and multiple agent types.  
It includes search-based agents, heuristic decision making, real-time planning, and adversarial/cooperative interactions.

## Main Components
- `SearchAgents/` — core implementation of agents, simulation logic, heuristics, and environment handling
- `agents/` — agent implementations such as A*, greedy, real-time, human, and adversarial agents
- `heuristics/` — heuristic strategies for evacuation planning
- `game_engine.py` / `simulation_engine.py` — execution and simulation flow
- `utils/` — parsing and helper utilities

## Algorithms and Behaviors
- A*
- Greedy heuristic agents
- Real-time A*
- Cooperative and adversarial agent behavior
- Simulation-based decision making

## Additional Resources
- `tests/` — test cases and example graph scenarios
- `docs/` — diagrams and documentation for agents and heuristics