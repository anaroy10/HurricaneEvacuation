import networkx as nx
from SearchAgents.agents.agent import Agent
from SearchAgents.consts import NUM_PEOPLE, HAS_KIT, EQUIP_TIME, UNEQUIP_TIME, KIT_SLOWER, WEIGHT
from SearchAgents.types.game_state import GameState


class GameEngine:
    def __init__(self, graph: nx.Graph, agents, game_type, deadline):
        self.world = graph
        self.agents = agents
        self.game_type = game_type
        self.deadline = deadline
        self.clock = 0

        # Track when each agent will be free to move again (for multi-turn actions)
        self.busy_until = {0: 0, 1: 0}

    def run(self):
        print(f"\n--- Starting Game ({self.game_type}) | Deadline: {self.deadline} ---")

        while self.clock < self.deadline:
            # Check Global Game Over
            total_people = sum(nx.get_node_attributes(self.world, NUM_PEOPLE).values())
            if total_people == 0:
                print("\nGAME OVER: All people saved!")
                break

            # Process each agent
            for i, agent in enumerate(self.agents):
                # 1. Check if agent is busy
                if self.busy_until[i] > self.clock:
                    continue

                    # 2. Build State Snapshot for AI
                state = self._build_game_state()

                # 3. Ask Agent for Decision
                action = agent.decide(state)

                # 4. Apply Action & Calculate Duration
                duration = self._apply_action(agent, action)

                # 5. Set Busy Timer
                self.busy_until[i] = self.clock + duration

                print(f"Time {self.clock}: Agent {i} decided '{action}' (Duration: {duration})")

            self.clock += 1

        self._print_results()

    def _build_game_state(self):
        """ Wraps current world into a clean GameState for the AI """
        agents_data = []
        for a in self.agents:
            agents_data.append({
                'pos': a.get_current_node(),
                'score': a.get_people_saved(),
                'has_kit': a.get_is_hold_kit()
            })
        return GameState(self.world, agents_data, self.clock)

    def _apply_action(self, agent: Agent, action_str: str):
        """ Applies action to real world and returns 'Cost' in turns """
        tokens = action_str.split()
        act = tokens[0]
        cost = 1  # Default

        if act == "traverse":
            dest = int(tokens[1])
            edge_data = self.world.get_edge_data(agent.get_current_node(), dest)
            w = edge_data.get(WEIGHT, 1)

            if agent.get_is_hold_kit():
                slower = self.world.graph.get(KIT_SLOWER, 1)
                cost = w * slower
            else:
                cost = w

            # Use FriendlyAgent logic to update position/people
            agent.traverse(dest)

        elif act == "equip":
            cost = self.world.graph.get(EQUIP_TIME, 1)
            agent.equip_kit()

        elif act == "unequip":
            cost = self.world.graph.get(UNEQUIP_TIME, 1)
            agent.unequip_kit()

        return cost

    def _print_results(self):
        print("\n--- Final Results ---")
        s0 = self.agents[0].get_people_saved()
        s1 = self.agents[1].get_people_saved()
        print(f"Agent 0 Saved: {s0}")
        print(f"Agent 1 Saved: {s1}")

        if self.game_type == 'adversarial':
            print(f"Final Score (0 - 1): {s0 - s1}")
        elif self.game_type == 'fully-cooperative':
            print(f"Final Team Score: {s0 + s1}")