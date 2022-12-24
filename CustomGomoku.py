import numpy as np
from copy import deepcopy
from typing import Tuple, Dict
from tqdm import tqdm
from collections import deque


class Gomoku:
    
    '''
    Gym-like environmet for Chienese game Gomoku (a.k.a. Five-In-Row)
    '''
    def __init__(self, height=10, width=10, len_to_win=5, current_player=1):
        
        self.height = height
        self.width = width
        self.len_to_win = len_to_win
        
        # Board for game.
        # Possible values 0, 1, 2
        # 0 - empty field
        # 1 - chip of 1st player
        # 2 - chip of 2nd player
        self.state = np.zeros((height, width))
        
        # Player's turn
        # 1 - turn of 1st player
        # 2 - turn of 2nd player
        self.current_player = current_player
        
    def _check_winner(self):
        
        # Exclude first and last self.len_to_win // 2 rows and columns.
        # Will check it later
        for i in range(self.len_to_win // 2, self.height - self.len_to_win // 2):
            for j in range(self.len_to_win // 2, self.width - self.len_to_win // 2):
                
                # Check diagonals
                main_d, side_d = 0, 0
                
                for k in range(-(self.len_to_win // 2), self.len_to_win // 2 + 1):
                    
#                     print(self.width, self.len_to_win // 2,i,j)
                    if self.state[i + k][j + k] == self.current_player:
                        
                        main_d += 1
                        
                    if self.state[i - k][j + k] == self.current_player:
                        
                        side_d += 1
                        
                # Check, is there a line with length self.len_to_win
                
                if self.len_to_win in (main_d, side_d,
                                       np.sum(self.state[i - self.len_to_win: i + self.len_to_win + 1] == self.current_player),
                                       np.sum(self.state[j - self.len_to_win: j + self.len_to_win + 1] == self.current_player)):
                    return True, self.current_player
                
            # Check first and last self.len_to_win rows
            for i in range(self.len_to_win // 2):
                
                for j in range(self.width - self.len_to_win + 1):
                    
                    if self.len_to_win in (
                        np.sum(self.state[i, j:j + self.len_to_win + 1] == self.current_player),
                        np.sum(self.state[self.height - i - 1, j:j + self.len_to_win + 1] == self.current_player)
                    ):
                        
                        return True, self.current_player
                    
            # Check first and last self.len_to_win columns
            for j in range(self.len_to_win // 2):
                
                for i in range(self.height - self.len_to_win + 1):
                    
                    if self.len_to_win in (
                        np.sum(self.state[i:i + self.len_to_win + 1, j] == self.current_player),
                        np.sum(self.state[i:i + self.len_to_win + 1, self.width - j - 1] == self.current_player)
                    ):
                        
                        return True, self.current_player
                    
        return False, -1
                    
    
    def _get_reward(self):
        
        '''Return rewards both for the 1st and for 2nd player'''
        
        flag, player = self._check_winner()
        
        if flag:
            
            rewards = [-1, -1]
            
            rewards[self.current_player - 1] = 1
            
            return rewards, True
        
        return [0, 0], False
    
    def available_actions(self):
        
        # Rows and columns
        return list(zip(*np.where(self.state == 0)))
        
    def reset(self, current_player=1):
        
        '''Start the new game from initial position'''
        
        self.state = np.zeros((self.height, self.width))
        self.current_player = current_player
        
        return self.state
    
    def step(self, action: Tuple):
        
        # Action is a tuple with (i, j) coordinates, 
        # where the current player place his chip
        
        self.state[action] = self.current_player
        rewards, is_done = self._get_reward()
        number_of_available_actions = self.height * self.width - len(self.available_actions())
        
        self.current_player = 2 - self.current_player + 1
        
        if is_done or number_of_available_actions == 0:
            
            # Terminal state
            return self.state, rewards, True
        
        return self.state, rewards, False
        
    def render(self):
        
        print(*self.state, sep='\n')
