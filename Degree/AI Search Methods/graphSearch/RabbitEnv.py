"""
This file contains the environment for the rabbit problem.
"""

from graphSearch.GraphSearch import BreadthFirstSearch

class RabbitEnv:
    def __init__(self):
        self.start = ['R','R','_','L','L']
        self.goal = ['L','L','_','R','R']
    
    def MoveGen(self, state):
        """
        This function generates the possible moves from the current state
        
        Args:
            state (List): Current state
            
        Returns:
            List: List of possible moves
        """
        moves = []
        i = 0
        for i in range(5):
            if state[i] == '_':
                break
        r_pos = [i-1, i-2]
        l_pos = [i+1, i+2]
        for j in r_pos:
            if j >= 0 and j <= 4 and state[j] == 'R':
                new_state = state.copy()
                new_state[i] = 'R'
                new_state[j] = '_'
                moves.append(new_state)
        for j in l_pos:
            if j >= 0 and j <= 4 and state[j] == 'L':
                new_state = state.copy()
                new_state[i] = 'L'
                new_state[j] = '_'
                moves.append(new_state)
        return moves
    
    def GoalTest(self, state):
        """
        This function checks if the current state is the goal state
        
        Args:
            state (List): Current state
            
        Returns:
            bool: True if the state is the goal state, False otherwise
        """
        return state == self.goal

def main():
    env = RabbitEnv()
    #print(env.MoveGen(['R','L','L','_','R']))
    #path = BreadthFirstSearch(env.start, env.MoveGen, env.GoalTest)
    #print("Final Path", path)
    #print("Length of Path", len(path))
    uniqueStates = [env.start]
    encounteredMoves = [env.start]
    while encounteredMoves != []:
        currentMove = encounteredMoves.pop(0)
        moves = env.MoveGen(currentMove)
        for move in moves:
            if move not in uniqueStates:
                uniqueStates.append(move)
                encounteredMoves.append(move)
    print("Unique States", uniqueStates)
    print("Number of Unique States", len(uniqueStates))
    