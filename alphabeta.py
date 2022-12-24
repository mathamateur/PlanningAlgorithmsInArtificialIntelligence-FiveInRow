import numpy as np
from copy import deepcopy
from typing import Tuple, Dict
from tqdm import tqdm
from collections import deque
from CustomGomoku import Gomoku


class Node:

    def __init__(self, env, heuristic_value_fn, depth=0, is_terminal=False):

        self.depth = depth
        self.env = env
        self.heuristic_value_fn = heuristic_value_fn
        self.is_terminal = is_terminal
    
    def get_heuristic_value(self):
        
        return self.heuristic_value_fn(self.env)

    
class AlphaBetaPlayer:
    
    def __init__(self, max_depth):
        
        self.max_depth = max_depth
        self.player = None
        
    def set_player_ind(self, p):
        self.player = p
        
    def __str__(self):
        return "AlphaBeta {}".format(self.player)
        
    def heuristic_value_fn(self, env):
        '''
        Calculate the longest line for each player which 
        doesn't contain chips of the opponent.
        '''
        longest_line = [0, 0]

        for player in range(1, 3):

            # Check main and side diaogonals

            for i in range(env.len_to_win // 2, env.height - env.len_to_win // 2):
                for j in range(env.len_to_win // 2, env.width - env.len_to_win // 2):

                    main_d, side_d = 0, 0

                    for k in range(-(env.len_to_win // 2), env.len_to_win // 2 + 1):
                        
                        if env.state[i + k][j + k] == player:

                            main_d += 1

                        elif env.state[i + k][j + k] == 2 - player + 1:

                            main_d = -10

                        if env.state[i - k][j + k] == player:

                            side_d += 1

                        elif env.state[i - k][j + k] == 2 - player + 1:

                            side_d = -10

                    longest_line[player - 1] = max([longest_line[player - 1], main_d, side_d])

            # Check rows

            for i in range(env.height):
                for j in range(env.width - env.len_to_win + 1):

                    row = 0 
                    for k in range(env.len_to_win - 1):

                        if env.state[i][j + k] == player:

                            row += 1

                        elif env.state[i][j + k] == 2 - player + 1:

                            row = -10

                    longest_line[player - 1] = max(longest_line[player - 1], row)

            # Check columns

            for i in range(env.height - env.len_to_win + 1):
                for j in range(env.width):

                    column = 0
                    for k in range(env.len_to_win - 1):

                        if env.state[i + k][j] == player:

                            column += 1

                        elif env.state[i + k][j] == 2 - player + 1:

                            column = -10

                    longest_line[player - 1] = max(longest_line[player - 1], column)

        max_player_proxy, min_player_proxy = longest_line

        if max_player_proxy == env.len_to_win:
            return 1
        elif min_player_proxy == env.len_to_win:
            return -1

        # convert score to [-1, 1] range (actually to smaller range, to make actual win more valuable)
        return (max_player_proxy - min_player_proxy) / env.len_to_win

    def _ndarray_to_tuple(self, ndarray):

        return tuple(map(tuple, ndarray))


    def alphabeta(self, env, node, depth, alpha, beta, maximizingPlayer, visited_states):
        
        if depth == 0:
            return node.get_heuristic_value()

        if maximizingPlayer:

            value = -float('inf')

            for action in env.available_actions():
                
                env_copy = deepcopy(env)

                new_state, reward, is_terminal = env_copy.step(action)
        
                new_state_tuple = self._ndarray_to_tuple(new_state)

                if new_state_tuple not in visited_states:

                    visited_states.add(new_state_tuple)

                    new_node = Node(env_copy, depth=depth - 1, heuristic_value_fn=self.heuristic_value_fn)

                    if is_terminal:

                        node.is_terminal = True

                        return reward[0] 

                    score = self.alphabeta(env_copy, new_node, depth - 1, alpha, beta, False, visited_states)
                    
                    value = max(value, score)

                if value >= beta:

                    break

                alpha = max(alpha, value)

            return value

        else:

            value = float('inf')

            for action in env.available_actions():

                env_copy = deepcopy(env)
                
                new_state, reward, is_terminal = env_copy.step(action)

                new_state_tuple = self._ndarray_to_tuple(new_state)

                if new_state_tuple not in visited_states:

                    visited_states.add(new_state_tuple)

                    new_node = Node(env_copy, depth=depth - 1, heuristic_value_fn=self.heuristic_value_fn)

                    if is_terminal:

                        node.is_terminal = True

                        return -reward[1] 
                    
                    score = self.alphabeta(env_copy, new_node, depth - 1, alpha, beta, True, visited_states)

                    value = min(value, score)

                if value <= alpha:

                    break

                beta = min(beta, value)

            return value
        
    def get_action(self, board):
        
        # Assumption that AlphaBeta player always moves first
        
        board_current_state = board.current_state()
        
        height, width = board_current_state[0].shape
        
        env = Gomoku(height=height, width=width)
        
        env.state = board_current_state[0] + board_current_state[1] * 2 
        
        actions = env.available_actions()
        
        best_action = actions[0]
        
        best_value = -1
        
        for action in actions:
            
            new_env = deepcopy(env)
            
            root = Node(env, heuristic_value_fn=self.heuristic_value_fn)
            
            visited_states = set()
            
            value = self.alphabeta(env, root, self.max_depth, -np.inf, np.inf, maximizingPlayer=True, visited_states=visited_states)
            
            if value > best_value:
                
                best_action = action
                
                best_value = value
                
        return best_action
            
