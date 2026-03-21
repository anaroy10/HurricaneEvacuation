# HurricaneEvacuation

A Python project for hurricane evacuation planning, search, simulation, and decision-making under uncertainty.

Developed in collaboration with **Lior Vinman**.

## Subprojects

### StormRunner
Graph-based hurricane evacuation framework that includes search, simulation, and multi-agent decision making.  
Implements A*, greedy heuristics, real-time search, and adversarial/cooperative agent behavior.

### FloodPath
Belief-state MDP implementation for hurricane routing under uncertainty.  
Models a stochastic shortest-path problem and computes policies using value iteration.

## Repository Structure

- `StormRunner/` — search, simulation, and multi-agent evacuation planning
- `FloodPath/` — planning under uncertainty using belief-state MDPs
