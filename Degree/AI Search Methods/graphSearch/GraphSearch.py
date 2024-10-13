"""
This file contains the implementation of the Graph Search algorithms
Breadth First Search
Depth First Search
Depth Limited Search
Iterative Deepening Search - N
Iterative Deepening Search - C
"""

from graphSearch.GeneralSearch import graph
import sys

def RemoveSeen(nodeList, Open, Closed):
    """
    Auxillary function that
    Removes nodes that have already been seen
    
    Args:
        nodeList (List): List of nodes
        Open (List): List of nodes that are open
        Closed (List): List of nodes that are closed

    Returns:
        List: List of nodes that have not been seen
    """
    if nodeList is None:
        return []
    nodeCheck = [x[0] for x in nodeList]
    open = [x[0] for x in Open]
    closed = [x[0] for x in Closed]
    nodes = []
    for i in range(len(nodeCheck)):
        if nodeCheck[i] not in open and nodeCheck[i] not in closed:
            nodes.append(nodeList[i])
    return nodes

def MakePairs(nodeList, parent):
    """
    Auxillary function that
    Makes pairs of nodes and their parents
    
    Args:
        nodeList (List): List of nodes
        parent (Any): Parent node

    Returns:
        List: List of pairs of nodes and their parents
    """
    if nodeList is None:
        return []
    return [(node, parent) for node in nodeList]

def SkipTo(parent, nodePairs, kind = "N", depth = 0):
    """
    Auxillary function that
    Skips to a parent node in a list of node pairs
    
    Args:
        parent (Any): Parent node
        nodePairs (List): List of node pairs

    Returns:
        List: List of node pairs starting from the parent node
    """
    i = 0
    if kind == "C":
        while i < len(nodePairs):
            if nodePairs[i][0] == parent and nodePairs[i][2] == depth:
                return nodePairs[i:]
            i += 1
    else:
        while i < len(nodePairs):
            if nodePairs[i][0] == parent:
                return nodePairs[i:]
            i += 1

def ReconstructPath(nodePair, Closed, kind = "G"):
    """
    Auxillary function that
    Reconstructs the path from the start node to the goal node
    
    Args:
        nodePair (List): List of node and its parent
        Closed (List): List of closed nodes

    Returns:
        List: List of nodes from the start node to the goal node
    """
    if kind == "G":
        # For BFS and DFS
        node, parent = nodePair
        path = [node]
        while parent is not None:
            path.insert(0, parent)
            Closed = SkipTo(parent, Closed)
            _, parent = Closed[0]
        return path
    # For DLS and IDS
    node, parent, depth = nodePair
    path = [node]
    while parent is not None:
        path.insert(0, parent)
        Closed = SkipTo(parent, Closed) if kind == "N" else SkipTo(parent, Closed, "C", depth-1)
        _, parent, depth = Closed[0]
    return path

def DepthFirstSearch(S, MoveGen, GoalTest):
    """
    This is a depth first search algorithm, may not be optimal
    
    Args:
        S (Any): Starting node
        MoveGen (function): Function that generates possible moves
        GoalTest (function): Function that tests if the goal has been reached

    Returns:
        Any: The path to goal node if found, otherwise an empty list
    """
    Open = [(S, None)]
    Closed = []
    while Open:
        N, parent = Open.pop(0)
        print("Choice:", N)
        print("Parent:", parent)
        if GoalTest(N):
            return ReconstructPath((N, parent), Closed)
        Closed = [(N, parent)] + Closed
        children = MoveGen(N)
        newNodes = RemoveSeen(children, Open, Closed)
        newPairs = MakePairs(newNodes, N)
        Open = newPairs + Open
        print("Open:", Open)
        print("Closed:", Closed)
    return []

def BreadthFirstSearch(S, MoveGen, GoalTest):
    """
    This is a breadth first search algorithm, optimal
    
    Args:
        S (Any): Starting node
        MoveGen (function): Function that generates possible moves
        GoalTest (function): Function that tests if the goal has been reached

    Returns:
        Any: The path to goal node if found, otherwise an empty list
    """
    Open = [(S, None)]
    Closed = []
    while Open:
        N, parent = Open.pop(0)
        print("Choice:", N)
        print("Parent:", parent)
        if GoalTest(N):
            return ReconstructPath((N, parent), Closed)
        Closed = [(N, parent)] + Closed
        children = MoveGen(N)
        newNodes = RemoveSeen(children, Open, Closed)
        newPairs = MakePairs(newNodes, N)
        # This line is the only difference between BFS and DFS
        Open = Open + newPairs
        print("Open:", Open)
        print("Closed:", Closed)
    return []

def MakePairsDepth(nodeList, parent, depth):
    """
    Auxillary function that
    Makes pairs of nodes and their parents
    
    Args:
        nodeList (List): List of nodes
        parent (Any): Parent node
        depth (int): Depth of the node

    Returns:
        List: List of pairs of nodes and their parents
    """
    if nodeList is None:
        return []
    return [(node, parent, depth) for node in nodeList]

def DepthLimitedSearch(S, MoveGen, GoalTest, limit, kind="N"):
    """
    This is a depth limited search algorithm, may not be optimal
    
    Args:
        S (Any): Starting node
        MoveGen (function): Function that generates possible moves
        GoalTest (function): Function that tests if the goal has been reached
        limit (int): The depth limit

    Returns:
        int: The number of nodes expanded
        Any: The path to goal node if found, otherwise an empty list
    """
    count = 0
    Open = [(S, None, 0)]
    Closed = []
    while Open:
        N, parent, depth = Open.pop(0)
        print("Choice:", N, "Depth:", depth)
        print("Parent:", parent)
        if GoalTest(N):
            path = ReconstructPath((N, parent, depth), Closed, kind)
            return count, path
        Closed = [(N, parent, depth)] + Closed
        if depth < limit:
            children = MoveGen(N)
            newNodes = RemoveSeen(children, Open, Closed) if kind == "N" else RemoveSeen(children, Open, [])
            newPairs = MakePairsDepth(newNodes, N, depth+1)
            Open = newPairs + Open
            count += len(newPairs)
        print("Open:", Open)
        print("Closed:", Closed)
    return count, []

def IterativeDeepeningSearch(S, MoveGen, GoalTest, kind="N"):
    """
    This is an iterative deepening search algorithm, optimal
    
    Args:
        S (Any): Starting node
        MoveGen (function): Function that generates possible moves
        GoalTest (function): Function that tests if the goal has been reached

    Returns:
        Any: The path to goal node if found, otherwise an empty list
    """
    count = -1
    path = []
    depthBound = 0
    print("Starting with depth bound:", depthBound)
    previousCount = count
    count, path = DepthLimitedSearch(S, MoveGen, GoalTest, depthBound, kind)
    depthBound += 1
    while path == [] and previousCount != count:
        print("Starting with depth bound:", depthBound)
        previousCount = count
        count, path = DepthLimitedSearch(S, MoveGen, GoalTest, depthBound, kind)
        depthBound += 1
    return path


def main():
    nodes = ['S', 'A', 'B', 'C', 'D', 'E', 'G']
    start = 'S'
    goal = 'G'
    # This is Graded Assignment 2 graph
    edges = {'S': ['A', 'C'], 'A': ['B', 'C', 'S'], 'B': ['A', 'D', 'E'], 'C': ['A', 'D', 'S'], 'D': ['B','C','E','G'], 'E': ['B','D','G'], 'G': ['D','E']}
    G = graph(nodes, edges, start, goal)
    
    with open("Degree/AI Search Methods/graphSearch/GraphSearch.txt", "w") as f:
        sys.stdout = f
        print("Depth First Search")
        path = DepthFirstSearch(G.start, G.MoveGen, G.GoalTest)
        print("Result path:", path)
        print()
        print("Breadth First Search")
        path = BreadthFirstSearch(G.start, G.MoveGen, G.GoalTest)
        print("Result path:", path)
        print()
        print("Iterative Deepening Search - N")
        path = IterativeDeepeningSearch(G.start, G.MoveGen, G.GoalTest)
        print("Result path:", path)
        print()
        print("Iterative Deepening Search - C")
        path = IterativeDeepeningSearch(G.start, G.MoveGen, G.GoalTest, "C")
        print("Result path:", path)
        sys.stdout = sys.__stdout__
    
if __name__ == "__main__":
    main()