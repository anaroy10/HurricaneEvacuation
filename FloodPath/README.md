# FloodPath

Belief-state MDP project for hurricane evacuation under uncertainty.

## Overview
This project models hurricane routing as a stochastic shortest-path problem under partial uncertainty.  
It computes policies using value iteration over belief states.

## Main Files
- `main.py` — entry point
- `mdp_solver.py` — value iteration / policy computation
- `belief_state_mdp.py` / `belief_space_mdp.py` — belief-state modeling
- `graph_parser.py` — input parsing

## Inputs and Outputs
- `inputs/` — example input instances
- `outputs/` — example output files