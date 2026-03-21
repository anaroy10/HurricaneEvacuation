import copy
import networkx as nx
from SearchAgents.consts import FLOODED, HAS_KIT, NUM_PEOPLE, WEIGHT, EQUIP_TIME, UNEQUIP_TIME, KIT_SLOWER


class GameState:
    """
    A snapshot of the game world used by the AI to plan moves.
    Includes logic for action duration (costs) and global deadline.
    """

    def __init__(self, graph, agents_data, turn_count=0, deadline=100, busy_times=None):
        self.graph = graph
        self.agents_data = agents_data
        self.turn_count = turn_count
        self.deadline = deadline
        # Tracks how many turns each agent is stuck doing an action [agent_0_busy, agent_1_busy]
        self.busy_times = busy_times if busy_times is not None else [0, 0]

    def get_legal_actions(self, agent_id):
        # 1. If agent is busy processing a previous action, they MUST wait.
        if self.busy_times[agent_id] > 0:
            return ["no-op"]

        actions = []
        agent = self.agents_data[agent_id]
        curr = agent['pos']
        has_kit = agent['has_kit']

        # Traverse
        for nbr in self.graph.neighbors(curr):
            edge = self.graph.get_edge_data(curr, nbr)
            # Check for flooding constraint
            if edge.get(FLOODED, False) and not has_kit:
                continue
            actions.append(f"traverse {nbr}")

        # Equip / Unequip
        node_has_kit = self.graph.nodes[curr].get(HAS_KIT, False)
        if not has_kit and node_has_kit:
            actions.append("equip")
        if has_kit:
            actions.append("unequip")

        # Always allow waiting (unless forced, but here it's a choice)
        actions.append("no-op")
        return actions

    def generate_successor(self, agent_id, action):
        """
        Returns a NEW GameState with the action applied.
        Calculates cost (duration) and updates busy timers.
        Advances the clock only after Agent 1 (the second player) moves.
        """
        new_graph = self.graph.copy()
        new_agents = copy.deepcopy(self.agents_data)
        new_busy = list(self.busy_times)

        agent = new_agents[agent_id]
        cost = 1  # Minimum cost for no-op

        # --- 1. Apply Logic & Calculate Cost ---
        if action.startswith("traverse"):
            dest = int(action.split()[1])
            edge_data = new_graph.get_edge_data(agent['pos'], dest)
            w = edge_data.get(WEIGHT, 1)

            # Update Position immediately (GameEngine does this too)
            agent['pos'] = dest

            # Calculate Cost based on Kit
            if agent['has_kit']:
                slower = new_graph.graph.get(KIT_SLOWER, 1)
                cost = w * slower
            else:
                cost = w

            # Prediction: Pick up people immediately upon arrival logic
            ppl = new_graph.nodes[dest].get(NUM_PEOPLE, 0)
            if ppl > 0:
                agent['score'] += ppl
                new_graph.nodes[dest][NUM_PEOPLE] = 0

        elif action == "equip":
            cost = new_graph.graph.get(EQUIP_TIME, 1)
            agent['has_kit'] = True
            new_graph.nodes[agent['pos']][HAS_KIT] = False

        elif action == "unequip":
            cost = new_graph.graph.get(UNEQUIP_TIME, 1)
            agent['has_kit'] = False
            new_graph.nodes[agent['pos']][HAS_KIT] = True

        elif action == "no-op":
            cost = 1

        # --- 2. Update Busy Timer for Current Agent ---
        # The agent is busy for (Cost - 1) turns *after* this current turn.
        # If cost is 1, busy becomes 0 (free next turn).
        new_busy[agent_id] = cost - 1

        # --- 3. Handle Time Passing ---
        # We assume strict turn order: Agent 0 -> Agent 1 -> Clock Tick.
        # If the acting agent is 1, we treat this as the end of a time unit.
        new_turn_count = self.turn_count

        if agent_id == 1:
            new_turn_count += 1
            # Decrement busy timers for EVERYONE because 1 clock tick passed
            for i in range(2):
                new_busy[i] = max(0, new_busy[i] - 1)

        return GameState(new_graph, new_agents, new_turn_count, self.deadline, new_busy)

    def is_game_over(self):
        # Game ends if people are saved OR deadline reached
        all_saved = sum(nx.get_node_attributes(self.graph, NUM_PEOPLE).values()) == 0
        time_up = self.turn_count >= self.deadline
        return all_saved or time_up

    # Standard getters...
    def get_score(self, agent_id):
        return self.agents_data[agent_id]['score']

    def get_agent_pos(self, agent_id):
        return self.agents_data[agent_id]['pos']

    def agent_has_kit(self, agent_id):
        return self.agents_data[agent_id]['has_kit']

    def has_people(self, node):
        return self.graph.nodes[node].get(NUM_PEOPLE, 0) > 0

    def is_agent_busy(self, agent_id):
        return self.busy_times[agent_id] > 0