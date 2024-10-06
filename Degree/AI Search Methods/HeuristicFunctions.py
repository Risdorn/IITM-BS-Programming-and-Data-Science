class EightTiles:
    def __init__(self, goal, start):
        self.start = start
        self.goal = goal
    
    def heuristic1(self, state):
        # Number of misplace tiles except 0
        return sum([1 for i in range(9) if state[i] != self.goal[i] and state[i] != 0])

    def heuristic2(self, state):
        # Manhattan distance except 0
        return sum([abs(i % 3 - self.goal.index(state[i]) % 3) + abs(i // 3 - self.goal.index(state[i]) // 3) for i in range(9) if state[i] != 0])
    
    def heuristic3(self, state):
        # Number of correctly placed tiles except 0
        return sum([1 for i in range(9) if state[i] == self.goal[i] and state[i] != 0])
    
    def heuristic4(self, state):
        # h3 - h1
        return self.heuristic3(state) - self.heuristic1(state)
    
    def MoveGen(self, state):
        # This will return new states that can be reached from the current state
        moves = []
        i = state.index(0)
        if i % 3 > 0: moves.append(state[:i-1] + [0, state[i-1]] + state[i+1:])
        if i % 3 < 2: moves.append(state[:i] + [state[i+1], 0] + state[i+2:])
        if i // 3 > 0: moves.append(state[:i-3] + [0] + state[i-2:i] + [state[i-3]] + state[i+1:])
        if i // 3 < 2: moves.append(state[:i] + [state[i+3]] + state[i+1:i+3] + [0] + state[i+4:])
        return moves

def eightTilesTest():
    goal = [1,2,3,8,0,4,7,6,5]
    start = [2,8,3,1,6,4,7,0,5]
    eightTiles = EightTiles(goal, start)
    A = [5,4,3,6,0,2,7,8,1]
    B = [5,4,3,6,2,0,7,8,1]
    C = [5,4,3,0,6,2,7,8,1]
    D = [5,4,3,6,7,2,0,8,1]
    E = [5,0,3,6,4,2,7,8,1]
    states = [goal, A, B, C, D, E]
    print("Order of states: goal, A, B, C, D, E")
    print("Heuristic 1:")
    for state in states:
        print(eightTiles.heuristic1(state), end=" ")
    print("\nHeuristic 2:")
    for state in states:
        print(eightTiles.heuristic2(state), end=" ")
    print("\nHeuristic 3:")
    for state in states:
        print(eightTiles.heuristic3(state), end=" ")
    print("\nHeuristic 4:")
    for state in states:
        print(eightTiles.heuristic4(state), end=" ")
    ANieghbors = eightTiles.MoveGen(A)
    print("\nNeighbors of A:")
    # H2 calculation
    for state in ANieghbors:
        print(state, eightTiles.heuristic2(state))

class BlockTower:
    def __init__(self, start, goal):
        # State is a dictionary like [[A,E,B,C,D], [F], []]
        # last element corresponds to table, so its the bottom level
        self.start = start
        self.goal = goal
    
    def findBlock(self, block):
        # Find indices of a block in the goal state
        for i in range(len(self.goal)):
            for j in range(len(self.goal[i])):
                if self.goal[i][j] == block:
                    return i, j
    
    def heuristic1(self, state):
        # add 1, if a block is on the correct block/table, otherwise, subtract 1.
        total = 0
        for i in range(len(state)):
            for j in range(len(state[i])):
                block = state[i][j]
                # Find the block in the goal state
                m, n = self.findBlock(block)
                if n == 0 and j == 0: total += 1
                elif n == 0 and j != 0: total -= 1
                elif n != 0 and j == 0: total -= 1
                elif state[i][j-1] == self.goal[m][n-1]: total += 1
                else: total -= 1
                #print(block, total, i, j, m, n)
        return total
                    

    def heuristic2(self, state):
        # for a block at level n, add n if the block rests on the correct structure below it, otherwise, subtract n.
        total = 0
        for i in range(len(state)):
            for j in range(len(state[i])):
                block = state[i][j]
                # Find the block in the goal state
                m, n = self.findBlock(block)
                if n == 0 and j == 0: total += 1
                elif n == 0 and j != 0: total -= j+1
                elif n != 0 and j == 0: total -= 1
                elif state[i][j-1] == self.goal[m][n-1]: total += j+1
                else: total -= j+1
        return total

    def heuristic3(self, state):
        # number of misplaced blocks, blocks not in final position.
        total = 0
        for i in range(len(state)):
            m = len(self.goal[i])
            if m == 0: 
                total += len(state[i])
                continue
            for j in range(len(state[i])):
                if j >= m: 
                    total += len(state[i]) - m
                    break
                if state[i][j] != self.goal[i][j]: total += 1
        return total
    
    def heuristic4(self, state):
        # number of correctly placed blocks, blocks in final position. 
        total = 0
        for i in range(len(state)):
            m = len(self.goal[i])
            if m == 0: continue
            for j in range(len(state[i])):
                if j >= m: break
                if state[i][j] == self.goal[i][j]: total += 1
        return total

def BlockTowerTest():
    goal = [["D","C","B","E","A"], ["F"], []]
    start = [["D","C","B","A"], ["F", "E"], []]
    blockTower = BlockTower(start, goal)
    P = [["D","C","B"], ["F", "E"], ["A"]]
    Q = [["D","C","B"], ["F","E","A"], []]
    R = [["D","C","B","A","E"], ["F"], []]
    T = [["D","C","B","A"], ["F"], ["E"]]
    W = [["D","C","B"], ["F"], ["A","E"]]
    X = [["D","C","B","E"], ["F"], ["A"]]
    states = [goal, start, P, Q, R, T, W, X]
    print("Order of states: goal, start, P, Q, R, T, W, X")
    print("Heuristic 1:")
    
    for state in states:
        print(blockTower.heuristic1(state), end=" ")
    print("\nHeuristic 2:")
    for state in states:
        print(blockTower.heuristic2(state), end=" ")
    print("\nHeuristic 3:")
    for state in states:
        print(blockTower.heuristic3(state), end=" ")
    print("\nHeuristic 4:")
    for state in states:
        print(blockTower.heuristic4(state), end=" ")
        
    #print(blockTower.heuristic2(W))

if __name__ == "__main__":
    #eightTilesTest()
    BlockTowerTest()