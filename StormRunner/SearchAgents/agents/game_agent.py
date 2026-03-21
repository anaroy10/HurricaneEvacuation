import math
from collections import deque
from typing import Tuple, Optional

import networkx as nx

from SearchAgents.agents.friendly_agent import FriendlyAgent
from SearchAgents.consts import FLOODED, NUM_PEOPLE


class GameAgent(FriendlyAgent):
    def __init__(self, g: nx.Graph, initial_node: int, agent_id: int, game_type: str, depth_limit: Optional[int] = 4):
        """
        Game playing agent, handles 3 game types as required.
        Decoupled from 'Busy' logic as GameState now handles duration constraints.

        Args:
            g (nx.Graph): The world graph
            initial_node (int): Start position
            agent_id (int): identifier of the agent (0 or 1)
            game_type (str): 'adversarial', 'semi-cooperative' or 'fully-cooperative'
            depth_limit (Optional[int]): how deep the Minimax search should go (default=4)
        """
        super().__init__(g, initial_node)

        if agent_id not in (0, 1):
            raise ValueError("agent ID should be '0' (first player) or '1' (second player) only")

        if game_type not in ("adversarial", "semi-cooperative", "fully-cooperative"):
            raise ValueError("supported game types: adversarial, semi-cooperative, fully-cooperative")

        self.agent_id = agent_id
        self.opponent_id = 1 - agent_id
        self.game_type = game_type
        self.depth_limit = depth_limit

    def decide(self, state):
        """
        Main decision method.
        Note: The GameEngine only calls this if the agent is NOT busy in the real world.
        """
        best_val = -math.inf
        best_action = "no-op"

        # Run the appropriate search algorithm
        if self.game_type == 'adversarial':
            # Alpha-Beta Pruning (Max vs Min)
            best_val, best_action = self.adversarial_search(
                state, self.depth_limit, -math.inf, math.inf, True
            )

        elif self.game_type == 'semi-cooperative':
            # Predictive Search (Max Me vs Max Them)
            best_val, best_action = self.semi_cooperative_search(
                state, self.depth_limit, True
            )

        elif self.game_type == 'fully-cooperative':
            # Maximax (Max Sum vs Max Sum)
            best_val, best_action = self.fully_cooperative_search(
                state, self.depth_limit, True
            )

        return best_action or "no-op"

    # =========================================================================
    # 1. ADVERSARIAL SEARCH (Zero-Sum: Me vs Enemy)
    # =========================================================================
    def adversarial_search(self, state, depth, alpha, beta, is_max_player):
        if depth == 0 or state.is_game_over():
            return self.evaluate_adversarial(state), None

        current_agent = self.agent_id if is_max_player else self.opponent_id
        legal_actions = state.get_legal_actions(current_agent)

        # If no actions (should not happen if "no-op" is always valid), return eval
        if not legal_actions:
            return self.evaluate_adversarial(state), "no-op"

        best_action = legal_actions[0]

        if is_max_player:
            max_eval = -math.inf
            for action in legal_actions:
                successor = state.generate_successor(current_agent, action)
                eval_score, _ = self.adversarial_search(successor, depth - 1, alpha, beta, False)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action

                # Pruning
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_action

        else:  # Minimizing Player (The Opponent)
            min_eval = math.inf
            for action in legal_actions:
                successor = state.generate_successor(current_agent, action)
                eval_score, _ = self.adversarial_search(successor, depth - 1, alpha, beta, True)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action

                # Pruning
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_action

    # =========================================================================
    # 2. SEMI-COOPERATIVE SEARCH (General-Sum: Me vs Selfish Opponent)
    # =========================================================================

    def semi_cooperative_search(self, state, depth, is_my_turn):
        player_to_move = self.agent_id if is_my_turn else self.opponent_id

        # --- Helper Functions for Tuple Comparison ---
        def heuristic_for(s, agent_idx: int) -> float:
            """Tiny tie-breaker: closer to remaining people is better."""
            try:
                pos = s.get_agent_pos(agent_idx)
            except Exception:
                return 0.0

            dist = self.bfs_distance_to_person(s, pos, agent_idx)
            if dist == 0:
                return 0.02
            if dist <= 0 or dist >= 999:
                return 0.0
            return 0.01 * (1.0 / dist)

        def eval_tuple(s) -> Tuple[float, float, float, float]:
            """ Returns (IS0, IS1, H0, H1). """
            is0 = float(s.get_score(0))
            is1 = float(s.get_score(1))
            h0 = heuristic_for(s, 0)
            h1 = heuristic_for(s, 1)
            return is0, is1, h0, h1

        def key_for_player(vec: Tuple[float, float, float, float], p: int):
            # Lexicographic sort: (My Score, Other Score, My Heuristic, Other Heuristic)
            if p == 0:
                return vec[0], vec[1], vec[2], vec[3]
            return vec[1], vec[0], vec[3], vec[2]

        def next_player(p: int) -> int:
            return 1 - p

        # --- Recursive MaxN ---
        def maxn(s, d: int, p: int):
            if d == 0 or s.is_game_over():
                return eval_tuple(s), None

            legal_actions = s.get_legal_actions(p)
            if not legal_actions:
                return eval_tuple(s), "no-op"

            best_vec = None
            best_action = legal_actions[0]

            for action in legal_actions:
                successor = s.generate_successor(p, action)
                # Recursively call for the next player
                vec, _ = maxn(successor, d - 1, next_player(p))

                # If this vector is better for player 'p' than the current best
                if best_vec is None or key_for_player(vec, p) > key_for_player(best_vec, p):
                    best_vec = vec
                    best_action = action

            return best_vec, best_action

        # --- Start Search ---
        vec, best_action = maxn(state, depth, player_to_move)

        # Convert tuple back to scalar for uniform return type (for debugging/logging compatibility)
        my_is = vec[self.agent_id]
        opp_is = vec[self.opponent_id]
        my_h = vec[2] if self.agent_id == 0 else vec[3]

        # Scalar score: (My Score * 1000) + Opponent Score + Heuristic
        return (my_is * 1000.0) + opp_is + my_h, best_action

    # =========================================================================
    # 3. FULLY COOPERATIVE SEARCH (Maximax: Me vs Teammate)
    # =========================================================================
    def fully_cooperative_search(self, state, depth, is_my_turn):
        if depth == 0 or state.is_game_over():
            return self.evaluate_fully_coop(state), None

        current_agent = self.agent_id if is_my_turn else self.opponent_id
        legal_actions = state.get_legal_actions(current_agent)

        if not legal_actions:
            return self.evaluate_fully_coop(state), "no-op"

        # Both agents act as MAX players for the total sum
        best_val = -math.inf
        best_action = legal_actions[0]

        for action in legal_actions:
            successor = state.generate_successor(current_agent, action)
            # Both players try to maximize the SAME score, so we keep looking for max
            val, _ = self.fully_cooperative_search(successor, depth - 1, not is_my_turn)

            if val > best_val:
                best_val = val
                best_action = action

        return best_val, best_action

    # =========================================================================
    # HEURISTIC EVALUATION FUNCTIONS
    # =========================================================================

    def _get_tie_breaker(self, state):
        """
        Calculates a small bonus based on distance to the nearest person.
        Smaller distance = Higher bonus.
        """
        my_pos = state.get_agent_pos(self.agent_id)
        dist = self.bfs_distance_to_person(state, my_pos, self.agent_id)

        if dist == 0:
            return 0.02  # Standing on a person (best case)
        if dist >= 999:
            return 0.0
        return 0.01 * (1.0 / dist)

    def evaluate_adversarial(self, state):
        # IS_me - IS_opp
        score = state.get_score(self.agent_id) - state.get_score(self.opponent_id)
        return score + self._get_tie_breaker(state)

    def evaluate_semi_coop(self, state):
        # NOT USED by the Semi-Coop Search (which uses Tuples),
        # but kept if you want to switch algorithms.
        score = (state.get_score(self.agent_id) * 1000) + state.get_score(self.opponent_id)
        return score + self._get_tie_breaker(state)

    def evaluate_fully_coop(self, state):
        # IS_me + IS_opp
        score = state.get_score(self.agent_id) + state.get_score(self.opponent_id)
        return score + self._get_tie_breaker(state)

    # =========================================================================
    # HELPERS
    # =========================================================================
    def bfs_distance_to_person(self, state, start_node, agent_idx):
        """
        Computes shortest path distance to any node containing people.
        Accounts for Flooded edges (requires kit).
        """
        # If people are at current location, distance is 0
        if state.has_people(start_node):
            return 0

        queue = deque([(start_node, 0)])
        visited = {start_node}
        has_kit = state.agent_has_kit(agent_idx)

        # Access the graph structure from the agent's memory
        graph = self._world

        while queue:
            curr, dist = queue.popleft()

            # Limit BFS depth to avoid performance hit
            if dist > 20:
                return 999

            for nbr in graph.neighbors(curr):
                if nbr in visited:
                    continue

                # Check edge constraints
                edge_data = graph.get_edge_data(curr, nbr)
                is_flooded = edge_data.get(FLOODED, False)

                if is_flooded and not has_kit:
                    continue

                # Check if this neighbor has people in the CURRENT state
                if state.has_people(nbr):
                    return dist + 1

                visited.add(nbr)
                queue.append((nbr, dist + 1))

        return 999  # No reachable people found