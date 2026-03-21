# Agents
* This project is an agent based environment simulation, which simulated the hurricane evacuation problem.
* Each agent can make decisions according to it's defined behaviour.
* Here we'll describe the existing agents in our simulation.
* This document presents each agent, and a flowchart for its initialization and decide().

## Agent's Architecture
* This is the current architecture of all agents in the system:

<img src="agents/AgentsUML.drawio.svg" alt="UML of all agents"/>


## Human Agent
* A friendly agent.
* Human directly-controlled agent, does not make any decision by itself, but receiving them as input from the user.

<img src="agents/HumanAgentFlow.drawio.svg" alt="human agent flowchart"/>


## Stupid Greedy Agent
* A friendly agent.
* Prefers shortest paths to the node that has the most people to rescue.
* Does not consider amphibian kits and always passes flooded edges.

<img src="agents/StupidGreedyAgentFlow.drawio.svg" alt="stupid greedy agent flowchart"/>

## Thief Agent
* A NON-friendly agent.
* Computes the shortest path to the nearst amphibian kit, then equips it.
* After equipping (or if it cannot traverse to a such kit), runs away from other agents.

<img src="agents/ThiefAgentFlow.drawio.svg" alt="stupid greedy agent flowchart"/>

## Heuristic Greedy Agent
* A friendly agent.
* Considers all actions - and decides according to greedy choice of the highest heuristic score.
* Read more about the used heuristic, [here](heuristics.md).

<img src="agents/HeuristicGreedyAgentFlow.drawio.svg" alt="heuristic greedy agent flowchart"/>

## A Star (optimal) Agent
* A friendly agent.
* Runs A* search over all operations until LIMIT of expansions, and takes always the optimal path.
* Read more about the used heuristic, [here](heuristics.md).

<img src="agents/AStarAgent.drawio.svg" alt="A* agent flowchart"/>

## Real Time A Star Agent
* A friendly agent.
* Runs limited A* search accordingly to LIMIT, and restarts from last node.
* Read more about the used heuristic, [here](heuristics.md).

<img src="agents/RealTimeAStarAgent.drawio.svg" alt="RTA* agent flowchart"/>

## Thief-Aware A Star Agent
* A friendly agent.
* Makes a prediction (inner simulation) of the thief agent and takes decision based on this, 
semi-suited (for a single thief agent).
* Read more about the used heuristic, [here](heuristics.md).

<img src="agents/ThiefAwareAStarAgent.drawio.svg" alt="thief aware A* agent flowchart"/>
