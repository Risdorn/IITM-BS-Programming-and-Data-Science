from GeneralSearch import graph
from HeuristicSearch import BestFirstSearch, HillClimbing, BeamSearch, BestNeighborDescent
from GraphSearch import BreadthFirstSearch
from TravellingSalesman import TravellingSalesman
import sys

class graphHeuristic(graph):
    def __init__(self, nodes, edges, coordinates, start, goal):
        """This is the constructor for the graphHeuristic class

        Args:
            nodes (List): List of nodes
            edges (Dictionary): Dictionary of edges
            coordinates (Dictionary): Dictionary of coordinates
            start (Any): Starting node
            goal (Any): Goal node
        """
        super().__init__(nodes, edges, start, goal)
        self.coordinates = coordinates
    
    def euclideanDistance(self, node, goal=None):
        """This function calculates the Euclidean distance between two nodes

        Args:
            node1 (Any): Node

        Returns:
            Float: Euclidean distance between given node and goal node
        """
        x1, y1 = self.coordinates[node]
        if goal == None: x2, y2 = self.coordinates[self.goal]
        else: x2, y2 = self.coordinates[goal]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def manhattanDistance(self, node, goal=None):
        """This function calculates the Manhattan distance between two nodes

        Args:
            node1 (Any): Node

        Returns:
            Float: Manhattan distance between the two nodes
        """
        x1, y1 = self.coordinates[node]
        if goal == None: x2, y2 = self.coordinates[self.goal]
        else: x2, y2 = self.coordinates[goal]
        return abs(x1 - x2) + abs(y1 - y2)

def main():
    # This graph is from Week 3 Practice Assignment 8
    nodes = ["S", "A", "B", "C", "D", "E", "F", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "G"]
    edges = {"S": ["C", "E", "I", "M"], "A":["B", "C", "D", "F"], "B":["A", "C", "D", "E", "H"], "C":["A", "B", "E", "S"],
             "D": ["A", "B", "F", "H", "L"], "E": ["B", "C", "H", "J", "S"], "F": ["A", "D", "G", "L"],
             "H": ["B", "D", "E", "J", "K"], "I": ["M", "R", "S"], "J": ["E", "H", "M"],
             "K": ["H", "L", "O"], "L": ["D", "F", "K", "O"], "M": ["I", "J", "P", "Q", "S"],
             "N": ["G", "O"], "O": ["K", "L", "N"], "P": ["M", "Q"], "Q": ["M", "P", "R"],
             "R": ["I", "Q"], "G": ["F", "N"]}
    coordinates = {"S": (75, 35), "A": (20, 60), "B": (40, 50), "C": (55, 60), "D": (25, 45), "E": (60, 40), "F": (10, 40),
                   "H": (40, 35), "I": (80, 15), "J": (55, 25), "K": (30, 25), "L": (15, 25), "M": (60, 15), "N": (10, 5),
                   "O": (25, 10), "P": (45, 10), "Q": (55, 0), "R": (75, 0), "G": (0, 15)}
    start = "S"
    goal = "G"
    GH = graphHeuristic(nodes, edges, coordinates, start, goal)
    with open("Degree/AI Search Methods/GraphHeuristicW3PA8.txt", "w") as file:
        sys.stdout = file
        print("MoveGen of B:", GH.MoveGen("B"))
        for node in nodes:
            print("Manhattan Distance from", node, "to G:", GH.manhattanDistance(node))
        print("Hill Climbing Search with graphHeuristic:")
        bestNode = HillClimbing(GH.start, GH.MoveGen, GH.manhattanDistance)
        print("Best Node:", bestNode)
        print()
        print("Best First Search with graphHeuristic:")
        path = BestFirstSearch(GH.start, GH.MoveGen, GH.GoalTest, GH.manhattanDistance)
        print("Path:", path)
        sys.stdout = sys.__stdout__
    
    # This graph is from Week 3 Graded Assignment
    nodes = ["S", "A", "B", "C", "D", "E", "F", "H", "I", "J", "K", "L", "M", "N", "O", "P", "G"]
    edges = {"S": ["B", "E", "J", "O"], "A":["B", "C", "E", "F"], "B":["A", "E", "S"], "C":["A", "D", "F"],
             "D": ["C", "G", "H"], "E": ["A", "B", "F", "I", "S"], "F": ["A", "C", "E", "K"],
             "H": ["D", "G", "L"], "I": ["E", "J"], "J": ["I", "O", "S"],
             "K": ["F", "M", "N"], "L": ["G", "H", "P"], "M": ["G", "K", "N", "P"],
             "N": ["K", "M", "O", "P"], "O": ["J", "S"], "P": ["L", "M", "N"], "G": ["D", "H", "L", "M"]}
    coordinates = {"S": (0, 50), "A": (60, 110), "B": (20, 100), "C": (90, 100), "D": (120, 90), "E": (40, 80), "F": (70, 70),
                   "H": (160, 70), "I": (50, 50), "J": (30, 40), "K": (80, 40), "L": (160, 40), "M": (110, 30), "N": (70, 10),
                   "O": (20, 0), "P": (130, 0), "G": (130, 50)}
    GH = graphHeuristic(nodes, edges, coordinates, start, goal)
    keys = {}
    for i in range(len(nodes)):
        keys[i] = nodes[i]
    graph = [[0 for i in range(len(nodes))] for j in range(len(nodes))]
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j: continue
            graph[i][j] = GH.euclideanDistance(nodes[i], nodes[j])
    TSP = TravellingSalesman(graph, 0, keys)
    with open("Degree/AI Search Methods/GraphHeuristicW3GA.txt", "w") as file:
        sys.stdout = file
        for node in nodes:
            print("Manhattan Distance from", node, "to G:", GH.manhattanDistance(node))
        print()
        print("Breadth First Search:")
        path = BreadthFirstSearch(GH.start, GH.MoveGen, GH.GoalTest)
        print("Path:", path)
        print()
        print("Best First Search with graphHeuristic:")
        path = BestFirstSearch(GH.start, GH.MoveGen, GH.GoalTest, GH.manhattanDistance)
        print("Path:", path)
        print()
        print("Hill Climbing Search with graphHeuristic:")
        bestNode = HillClimbing(GH.start, GH.MoveGen, GH.manhattanDistance)
        print("Best Node:", bestNode)
        print()
        print("Best Neighbor Descent with graphHeuristic:")
        bestNode = BestNeighborDescent(GH.start, GH.MoveGen, GH.manhattanDistance, len(GH.nodes))
        print("Best Node:", bestNode)
        print()
        print("Traveling Salesman Problem:")
        print("Nearest Neighbor Tour:", TSP.computeTour(), "Cost:", TSP.TourCost(TSP.computeTour()))
        print("Savings Tour:", TSP.savingsTour(0), "Cost:", TSP.TourCost(TSP.savingsTour(0)))
        print("Greedy Tour:", TSP.computeGreedyTour(), "Cost:", TSP.TourCost(TSP.computeGreedyTour()))
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()