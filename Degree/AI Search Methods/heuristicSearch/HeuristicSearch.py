"""
This file implements the following heuristic search algorithms:
1. Best First Search
2. Hill Climbing
3. Best Neighbor Search
"""

from heuristicSearch.HeuristicFunctions import EightTiles, BlockTower

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
    # Node, Parent, Cost
    return [(node[0], parent, node[1]) for node in nodeList]

def SkipTo(parent, nodePairs):
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
    while i < len(nodePairs):
        if nodePairs[i][0] == parent:
            return nodePairs[i:]
        i += 1

def ReconstructPath(nodePair, Closed):
    """
    Auxillary function that
    Reconstructs the path from the start node to the goal node
    
    Args:
        nodePair (List): List of node and its parent
        Closed (List): List of closed nodes

    Returns:
        List: List of nodes from the start node to the goal node
    """
    node, parent = nodePair
    path = [node]
    while parent is not None:
        path.insert(0, parent)
        Closed = SkipTo(parent, Closed)
        _, parent, _ = Closed[0]
    return path

def CalculateCost(nodeList, Heuristic):
    """
    Auxillary function that
    Calculates the cost of nodes
    
    Args:
        nodeList (List): List of nodes
        Heuristic (Function): Heuristic function

    Returns:
        List: List of nodes with their costs
    """
    if nodeList is None:
        return []
    return [(node, Heuristic(node)) for node in nodeList]

def BestFirstSearch(S, MoveGen, GoalTest, Heuristic, kind="min"):
    """
    Best First Search Algorithm
    Uses a heuristic to find the best node to expand

    Args:
        S (Any): Start node
        MoveGen (function): Function that generates possible moves from a node
        GoalTest (function): Function that checks if a node is the goal node
        Heuristic (function): Heuristic function
        kind (str, optional): Whether heuristic function is "max" problem or "min problem". Defaults to "min".

    Returns:
        List: List of nodes from the start node to the goal node
    """
    Open = [(S, None, Heuristic(S))]
    Closed = []
    while Open:
        N, parent, cost = Open.pop(0)
        print("Choice:", N)
        print("Parent:", parent)
        if GoalTest(N):
            return ReconstructPath((N, parent), Closed)
        Closed = [(N, parent, cost)] + Closed
        children = MoveGen(N)
        children = CalculateCost(children, Heuristic)
        newNodes = RemoveSeen(children, Open, Closed)
        newPairs = MakePairs(newNodes, N)
        # Now we will sort Open in ascending order of cost
        Open = newPairs + Open
        if kind == "min": Open.sort(key=lambda x: x[2])
        elif kind == "max": Open.sort(key=lambda x: x[2], reverse=True)
        else: 
            print("Invalid kind") 
            break
        print("Open:", Open)
        print("Closed:", Closed)
    return []

def betterHeuristic(heuristic1, heuristic2, kind = "min"):
    """
    Auxillary function that
    Compares two heuristics based on the kind of problem
    
    Args:
        heuristic1 (Any): Heuristic 1
        heuristic2 (Any): Heuristic 2
        kind (str, optional): Whether heuristic function is "max" problem or "min problem". Defaults to "min.
        
    Returns:
        Boolean: True if heuristic1 is better than heuristic2 else False
    """
    if kind == "min": return heuristic1 < heuristic2
    elif kind == "max": return heuristic1 > heuristic2
    return None

def HillClimbing(S, MoveGen, Heuristic, kind = "min"):
    """
    Hill Climbing Algorithm
    Moves to the best node in the neighbourhood

    Args:
        S (Any): Start node
        MoveGen (function): Function that generates possible moves from a node
        Heuristic (function): Heuristic function
        kind (str, optional): Whether heuristic function is "max" problem or "min problem". Defaults to "min".

    Returns:
        Any: Best node
    """
    i = 1
    N = S
    bestEver = N
    print("Iteration:", i, "Node:", N, "Heuristic:", Heuristic(N))
    i += 1
    moves = MoveGen(bestEver)
    moves = CalculateCost(moves, Heuristic)
    if kind == "min": moves.sort(key=lambda x: x[1])
    elif kind == "max": moves.sort(key=lambda x: x[1], reverse=True)
    else: 
        print("Invalid kind") 
        return []
    N = moves[0][0]
    while betterHeuristic(Heuristic(N), Heuristic(bestEver), kind):
        bestEver = N
        print("Iteration:", i, "Node:", N, "Heuristic:", Heuristic(N))
        moves = MoveGen(bestEver)
        moves = CalculateCost(moves, Heuristic)
        if kind == "min": moves.sort(key=lambda x: x[1])
        elif kind == "max": moves.sort(key=lambda x: x[1], reverse=True)
        else: 
            print("Invalid kind") 
            return []
        N = moves[0][0]
        i += 1
    print("Iteration:", i, "Node:", N, "Heuristic:", Heuristic(N))
    return bestEver

def BestNeighborDescent(S, MoveGen, Heuristic, nodeCount, kind = "min"):
    """
    Best Neighbor Descent Algorithm
    Moves to the best node in the neighbourhood, as long as the heuristic is not 0 and the node count is less than the node count

    Args:
        S (Any): Start node
        MoveGen (function): Function that generates possible moves from a node
        Heuristic (function): Heuristic function
        nodeCount (int): Number of nodes
        kind (str, optional): Whether heuristic function is "max" problem or "min problem". Defaults to "min".

    Returns:
        Any: Best node
    """
    i = 1
    N = S
    bestEver = N
    while Heuristic(bestEver) != 0 and i-1 < nodeCount:
        bestEver = N
        print("Iteration:", i, "Node:", N, "Heuristic:", Heuristic(N))
        moves = MoveGen(bestEver)
        moves = CalculateCost(moves, Heuristic)
        if kind == "min": moves.sort(key=lambda x: x[1])
        elif kind == "max": moves.sort(key=lambda x: x[1], reverse=True)
        else: 
            print("Invalid kind") 
            return []
        N = moves[0][0]
        i += 1
    print("Iteration:", i, "Node:", N, "Heuristic:", Heuristic(N))
    return bestEver

def RemoveDups(moves):
    # List of lists
    newMoves = []
    for move in moves:
        if move not in newMoves:
            newMoves.append(move)
    return newMoves

def MoveBeam(Open, MoveGen, Heuristic, kind = "min"):
    moves = []
    for node in Open:
        moves.extend(MoveGen(node[0]))
    moves = RemoveDups(moves)
    moves = CalculateCost(moves, Heuristic)
    if kind == "min": moves.sort(key=lambda x: x[1])
    elif kind == "max": moves.sort(key=lambda x: x[1], reverse=True)
    
    return moves

def GoalBeam(Open, GoalTest):
    """
    Auxillary function that
    Tests if the goal has been reached
    
    Args:
        Open (List): List of nodes
        GoalTest (function): Function that tests if the goal has been reached

    Returns:
        List: List of nodes that are the goal nodes
    """
    for node in Open:
        if GoalTest(node[0]):
            return node, True
    return None, False

def BeamSearch(S, MoveGen, GoalTest, Heuristic, beamWidth, kind = "min"):
    """
    Beam Search Algorithm
    Expands the best nodes in the neighbourhood, with a beam width of beamWidth

    Args:
        S (Any): Start node
        MoveGen (function): Function that generates possible moves from a node
        GoalTest (function): Function that tests if the goal has been reached
        Heuristic (function): Heuristic function
        beamWidth (int): Beam width
        kind (str, optional): Whether heuristic function is "max" problem or "min problem". Defaults to "min".

    Returns:
        Any: Best node
    """
    i = 1
    Open = [(S, Heuristic(S))]
    N = S
    bestEver = N
    print("Iteration:", i, "Node:", N, "Heuristic:", Heuristic(N))
    i += 1
    _, goal = GoalBeam(Open, GoalTest)
    if goal: return N
    Open = MoveBeam(Open, MoveGen, Heuristic, kind)[:beamWidth]
    print("Open:", Open)
    N = Open[0][0]
    while betterHeuristic(Heuristic(N), Heuristic(bestEver), kind):
        bestEver = N
        print("Iteration:", i, "Node:", N, "Heuristic:", Heuristic(N))
        _, goal = GoalBeam(Open, GoalTest)
        if goal: return _
        Open = MoveBeam(Open, MoveGen, Heuristic, kind)[:beamWidth]
        print("Open:", Open)
        N = Open[0][0]
        i += 1
    print("Iteration:", i, "Node:", N, "Heuristic:", Heuristic(N))
    return bestEver

def main():
    # EightTiles
    start = [1, 2, 3, 4, 5, 6, 7, 0, 8]
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    E = EightTiles(start, goal)
    print("Hill Climbing Search with EightTiles:")
    print(HillClimbing(E.start, E.MoveGen, E.heuristic2, "min"))
    # BlockTower
    goal = [["D","C","B","E","A"], ["F"], []]
    start = [["D","C","B","A"], ["F", "E"], []]
    B = BlockTower(start, goal)
    print("Hill Climbing Search with BlockTower:")
    print(HillClimbing(B.start, B.MoveGen, B.heuristic1, "max"))