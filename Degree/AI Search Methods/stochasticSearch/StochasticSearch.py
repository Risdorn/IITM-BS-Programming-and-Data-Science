"""
This module contains the implementation of the Stochastic Search algorithms.
1. Iterated Hill Climbing
2. Random Walk
3. Simulated Annealing
4. Ant Colony Optimization
"""
from heuristicSearch.HeuristicSearch import HillClimbing
from heuristicSearch.HeuristicFunctions import BlockTower
import numpy as np
import sys
import io

class BlockTowerRandom(BlockTower):
    def __init__(self, start, goal):
        super().__init__(start, goal)
        self.blocks = [j for i in start for j in i]
    
    def randomGenerator(self):
        # Generate a random state
        blocks = self.blocks.copy()
        blocks = list(np.random.permutation(blocks))
        indices = np.random.choice(range(len(blocks)), 2, replace=False)
        state = [blocks[:indices[0]], blocks[indices[0]:indices[1]], blocks[indices[1]:]]
        return state

def iteratedHillClimbing(randomGenerator, MoveGen, heuristic, iterations, kind = "min"):
    """Iterated Hill Climbing algorithm

    Args:
        randomGenerator (function): Random generator function
        MoveGen (function): Move generator function
        heuristic (function): Heuristic function
        iterations (int): Number of iterations
        kind (str, optional): Type of optimization. Defaults to "min".

    Returns:
        Any: Best node
    """
    bestNode = randomGenerator()
    for _ in range(iterations):
        node = randomGenerator()
        output = sys.stdout
        sys.stdout = io.StringIO()
        node = HillClimbing(node, MoveGen, heuristic, kind)
        sys.stdout = output
        if kind == "min" and heuristic(node) < heuristic(bestNode): bestNode = node
        elif kind == "max" and heuristic(node) > heuristic(bestNode): bestNode = node
    return bestNode

def randomWalk(start, MoveGen, heuristic, iterations, kind = "min"):
    """Random Walk algorithm

    Args:
        start (Any): Start node
        MoveGen (function): Move generator function
        heuristic (function): Heuristic function
        iterations (int): Number of iterations
        kind (str, optional): Type of optimization. Defaults to "min".
    
    Returns:
        Any: Best node
    """
    node = start
    bestNode = start
    for _ in range(iterations):
        moves = MoveGen(node)
        node = np.random.choice(range(len(moves)), 1)[0]
        node = moves[node]
        if kind == "min" and heuristic(node) < heuristic(bestNode): bestNode = node
        elif kind == "max" and heuristic(node) > heuristic(bestNode): bestNode = node
    return bestNode

def simulatedAnnealing(start, MoveGen, heuristic, T, alpha, kind = "min", epochs = 10, iterations = 50):
    """Simulated Annealing algorithm

    Args:
        start (Any): Start node
        MoveGen (function): Move generator function
        heuristic (function): Heuristic function
        T (float): Temperature
        alpha (float): Alpha
        kind (str, optional): Type of optimization. Defaults to "min".
        epochs (int, optional): Number of epochs. Defaults to 10.
        iterations (int, optional): Number of iterations. Defaults to 50.

    Returns:
        Any: Best node
    """
    # Inner function, to avoid code repetition
    def sigmoid(x, T):
        return 1/(1 + np.exp(-x/T))
    
    # Main function
    node = start
    bestNode = start
    for time in range(1, epochs):
        for _ in range(iterations):
            moves = MoveGen(node)
            nextNode = np.random.choice(range(len(moves)), 1)[0]
            nextNode = moves[nextNode]
            delta = heuristic(nextNode) - heuristic(node)
            # Explorative move
            if (delta < 0 and kind == "min") or np.random.rand() < sigmoid(delta, T): node = nextNode
            elif (delta > 0 and kind == "max") or np.random.rand() < sigmoid(-delta, T): node = nextNode
            # Exploitative move
            if kind == "min" and heuristic(node) < heuristic(bestNode): bestNode = node
            elif kind == "max" and heuristic(node) > heuristic(bestNode): bestNode = node
        T = alpha * T
    return bestNode

def main():
    # Test the Stochastic Search algorithms
    start = [["D","C","B","A"], ["F", "E"], []]
    goal = [["D","C","B","E","A"], ["F"], []]
    blocks = BlockTowerRandom(start, goal)
    state = blocks.randomGenerator()
    print("Goal State:", goal)
    print("Start State:", state)
    best = iteratedHillClimbing(blocks.randomGenerator, blocks.MoveGen, blocks.heuristic1, 1000, kind="max")
    print("Iterated Hill Climbing:", best)
    best = randomWalk(state, blocks.MoveGen, blocks.heuristic1, 1000, kind="max")
    print("Random Walk:", best)
    best = simulatedAnnealing(state, blocks.MoveGen, blocks.heuristic1, 100, 0.99, kind="max", epochs=10, iterations=50)
    print("Simulated Annealing:", best)

if __name__ == "__main__":
    main()