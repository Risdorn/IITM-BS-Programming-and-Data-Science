"""
This file defines two general search algorithms
Simple Search 1
Simple Search 2
"""

import random

def SimpleSearch1(S, MoveGen, GoalTest):
    """
    This is a simple search algorithm that picks points in any order
    Can get stuck in loops no guarantee of finding a solution
    Args:
        S (Any): Starting node
        MoveGen (function): Function that generates possible moves
        GoalTest (function): Function that tests if the goal has been reached
    """
    count = 0
    open = [S]
    while open:
        # Choose a random node
        N = random.choice(open)
        open.remove(N)
        if GoalTest(N):
            return N
        open.extend(MoveGen(N))
        count += 1
        if count > 100: break # To prevent infinite loops
    return "Failure"

def SimpleSearch2(S, MoveGen, GoalTest):
    """
    This is a simple search algorithm that picks points in any order but uses a set to keep track of visited nodes
    Solves the problem of getting stuck in loops
    Args:
        S (Any): Starting node
        MoveGen (function): Function that generates possible moves
        GoalTest (function): Function that tests if the goal has been reached
    """
    open = [S]
    closed = set()
    while open:
        N = random.choice(open)
        open.remove(N)
        if GoalTest(N):
            return N
        closed.add(N)
        open.extend([M for M in MoveGen(N) if M not in closed])
    return "Failure"

class graph:
    """
    This class defines a graph
    """
    def __init__(self, nodes, edges, start, goal):
        """This is the constructor for the graph class

        Args:
            nodes (List): List of nodes
            edges (Dictionary): Dictionary of edges
            start (Any): Starting node
            goal (Any): Goal node
        """
        self.nodes = nodes
        self.edges = edges
        self.start = start
        self.goal = goal
    
    def MoveGen(self, node):
        """This function generates possible moves from a node

        Args:
            node (Any): Node

        Returns:
            List: List of possible moves from the node
        """
        return self.edges[node]
    
    def GoalTest(self, node):
        """This function checks if node is the goal node

        Args:
            node (Any): Node

        Returns:
            Boolean: True if node is the goal node else False
        """
        return node == self.goal

def main():
    nodes = ["S", "A", "B", "C", "D", "G"]
    start = "S"
    goal = "G"
    edges = {"S": ["A", "B"], "A": ["C", "D"], "B": ["G"], "C": ["G"], "D": ["G"], "G": []} # Graph with no loops
    edges1 = {"S": ["A", "B"], "A": ["C", "D"], "B": [], "C": [], "D": [], "G": []} # Graph with no path to goal
    edges2 = {"S": ["A", "B"], "A": ["C", "D"], "B": ["C", "D"], "C": ["A"], "D": ["G"], "G": []} # Graph with loops SimpleSearch1 could get stuck in loop
    G = graph(nodes, edges, start, goal)
    G1 = graph(nodes, edges1, start, goal)
    G2 = graph(nodes, edges2, start, goal)
    print("Simple Search 1 with a graph with no loops:", SimpleSearch1(G.start, G.MoveGen, G.GoalTest))
    print("Simple Search 2 with a graph with no loops:", SimpleSearch2(G.start, G.MoveGen, G.GoalTest))
    print("Simple Search 1 with a graph with no path to goal:", SimpleSearch1(G1.start, G1.MoveGen, G1.GoalTest))
    print("Simple Search 2 with a graph with no path to goal:", SimpleSearch2(G1.start, G1.MoveGen, G1.GoalTest))
    print("Simple Search 1 with a graph with loops:", SimpleSearch1(G2.start, G2.MoveGen, G2.GoalTest))
    print("Simple Search 2 with a graph with loops:", SimpleSearch2(G2.start, G2.MoveGen, G2.GoalTest))

if __name__ == "__main__":
    main()