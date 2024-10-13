import numpy as np
from heuristicSearch.TravellingSalesman import TravellingSalesman

class TSPGenetic():
    def __init__(self, graph, start, keys):
        self.graph = graph
        self.start = start
        self.keys = keys
        self.reverse = {v:k for k,v in keys.items()}
    
    def MoveGen(self, state, visited = []):
        # Generate all possible moves from the current state
        moves = []
        for i in range(len(self.graph[state])):
            if i != state and i not in visited and self.graph[state][i] != 0:
                moves.append(i)
        return moves
    
    def GenerateCandidate(self, start):
        # Generate a candidate solution using the MoveGen function
        tour = [start]
        current = start
        while len(tour) < len(self.graph):
            moves = self.MoveGen(current, tour)
            current = np.random.choice(range(len(moves)), 1)[0]
            current = moves[current]
            tour.append(current)
        tour = [self.keys[i] for i in tour]
        return tour
    
    def adjacencyRepresentation(self, tour):
        # Convert the tour to an adjacency representation
        adj = [None]*len(tour)
        tour = [self.reverse[i] for i in tour]
        for i in range(len(tour)-1):
            adj[tour[i]] = tour[i+1]
        adj[tour[-1]] = tour[0]
        adj = [self.keys[i] for i in adj]
        return adj
    
    def ordinalRepresentation(self, tour):
        # Convert the tour to an ordinal representation
        ordinal = [None]*len(tour)
        tour = [self.reverse[i] for i in tour]
        converted = []
        for i in range(len(tour)):
            indice = tour[i]
            val = 0
            for j in range(len(converted)):
                if indice > converted[j]: val += 1
            ordinal[i] = indice - val + 1
            converted.append(indice)
        return ordinal
    
    def partiallyMappedCrossover(self, parent1, parent2, start, end):
        # Perform partially mapped crossover on two parents
        child1 = [None]*len(parent1)
        child2 = [None]*len(parent1)
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        for i in range(start, end):
            if parent2[i] not in child1:
                pos = i
                while True:
                    index = parent2.index(parent1[pos])
                    if child1[index] == None:
                        child1[index] = parent2[i]
                        break
                    else: pos = index
            if parent1[i] not in child2:
                pos = i
                while True:
                    index = parent1.index(parent2[pos])
                    if child2[index] == None:
                        child2[index] = parent1[i]
                        break
                    else: pos = index
        for i in range(len(child1)):
            if child1[i] == None: child1[i] = parent2[i]
            if child2[i] == None: child2[i] = parent1[i]
        return child1, child2
    
    def orderCrossover(self, parent1, parent2, start, end):
        # Perform order crossover on two parents
        child1 = [None]*len(parent1)
        child2 = [None]*len(parent1)
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        j, k = 0, 0
        for i in range(len(parent2)):
            flag1, flag2 = True, True
            if parent2[i] in child1: flag1 = False
            if parent1[i] in child2: flag2 = False
            if flag1: child1[j] = parent2[i]
            if flag2: child2[k] = parent1[i]
            while j < len(child1) and child1[j] != None: j += 1
            while k < len(child2) and child2[k] != None: k += 1
        return child1, child2
    
    def cycleCrossover(self, parent1, parent2):
        # Perform cycle crossover on two parents
        child1 = [None]*len(parent1)
        child2 = [None]*len(parent1)
        cycles = {}
        cities, i = 1, 0
        visited = [False]*len(parent1)
        while cities < len(parent1) and i < len(visited):
            cycle = [parent1[i]]
            visited[i] = True
            i = parent1.index(parent2[i])
            while parent1[i] != cycle[0]:
                cycle.append(parent1[i])
                visited[i] = True
                i = parent1.index(parent2[i])
            cycles[cities] = cycle
            cities += 1
            i = 0
            while i < len(visited) and visited[i]: i += 1
        
        for i in range(1, len(cycles)+1):
            cycle = cycles[i]
            if i%2 == 0:
                for j in range(len(cycle)):
                    p1indice = parent1.index(cycle[j])
                    p2indice = parent2.index(cycle[j])
                    child1[p2indice] = cycle[j]
                    child2[p1indice] = cycle[j]
            else:
                for j in range(len(cycle)):
                    p1indice = parent1.index(cycle[j])
                    p2indice = parent2.index(cycle[j])
                    child1[p1indice] = cycle[j]
                    child2[p2indice] = cycle[j]
        return child1, child2
    
    def singlePointCrossover(self, parent1, parent2, point):
        # Perform single point crossover on two parents
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2


def main():
    keys = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J", 10:"K", 11:"L", 12:"M", 13:"N", 14:"O"}
    graph = [[0, 50, 36, 28, 30, 72, 50], [50, 0, 82, 36, 58, 41, 71], [36, 82, 0, 50, 32, 92, 42], [28, 36, 50, 0, 22, 45, 36],
             [30, 58, 32, 22, 0, 61, 20], [72, 41, 92, 45, 61, 0, 61], [50, 71, 42, 36, 20, 61, 0]]
    TSP = TSPGenetic(graph, 1, keys)
    candidate = ['F','L','A','M','E','B','I','N','G','O','J','K','H','C','D']
    candidate1 = candidate.copy()
    item = candidate1.pop(0)
    candidate1.append(item)
    print("Representation of the tour:")
    print("Path Representation", candidate)
    adj = TSP.adjacencyRepresentation(candidate)
    print("Adjacency Representation", adj)
    ordi = TSP.ordinalRepresentation(candidate)
    print("Ordinal Representation", ordi)
    print()
    print("Path Representation", candidate1)
    ordi1 = TSP.ordinalRepresentation(candidate1)
    print("Ordinal Representation", ordi1)
    parent1 = ["F","L","A","M","E","B","I","N","G","O","J","K","H","C","D"]
    parent2 = ["B","N","O","I","M","J","A","L","K","E","H","C","F","D","G"]
    child1, child2 = TSP.partiallyMappedCrossover(parent1, parent2, 5, 10)
    print()
    print("Partially Mapped Crossover")
    print("Parent 1", parent1)
    print("Parent 2", parent2)
    print("Child 1", child1)
    print("Child 2", child2)
    print()
    print("Order Crossover")
    child1, child2 = TSP.orderCrossover(parent1, parent2, 5, 10)
    print("Child 1", child1)
    print("Child 2", child2)
    print()
    print("Cycle Crossover")
    child1, child2 = TSP.cycleCrossover(parent1, parent2)
    print("Child 1", child1)
    print("Child 2", child2)
    print()
    print("Single Point Crossover")
    ordinal1 = TSP.ordinalRepresentation(parent1)
    ordinal2 = TSP.ordinalRepresentation(parent2)
    child1, child2 = TSP.singlePointCrossover(ordinal1, ordinal2, 5)
    print("Child 1", child1)
    print("Child 2", child2)
    
    print("Representation of the tour:")
    tour1 = ["D","I","J","O","N","H","G","B","C","K","F","L","A","M","E"]
    tour2 = ["A","L","F","K","C","B","G","H","N","O","J","I","D","E","M"]
    print("Path Representation", tour1)
    adj = TSP.adjacencyRepresentation(tour1)
    print("Adjacency Representation", adj)
    ordi = TSP.ordinalRepresentation(tour1)
    print("Ordinal Representation", ordi)
    print()
    print("Path Representation", tour2)
    adj1 = TSP.adjacencyRepresentation(tour2)
    print("Adjacency Representation", adj1)
    ordi1 = TSP.ordinalRepresentation(tour2)
    print("Ordinal Representation", ordi1)
    
    tour3 = ['H','B','F','N','K','J','I','C','G','A','D','M','O','E','L']
    print("Partially Mapped Crossover")
    child1, child2 = TSP.partiallyMappedCrossover(tour1, tour3, 5, 10)
    print("Child 1", child1)
    print("Child 2", child2)
    print()
    print("Order Crossover")
    child1, child2 = TSP.orderCrossover(tour1, tour3, 5, 10)
    print("Child 1", child1)
    print("Child 2", child2)
    print()
    print("Cycle Crossover")
    child1, child2 = TSP.cycleCrossover(tour1, tour3)
    print("Child 1", child1)
    print("Child 2", child2)
    print()
    print("Single Point Crossover")
    ordinal1 = TSP.ordinalRepresentation(tour1)
    ordinal2 = TSP.ordinalRepresentation(tour3)
    child1, child2 = TSP.singlePointCrossover(ordinal1, ordinal2, 7)
    print("Ordinal 1", ordinal1)
    print("Ordinal 2", ordinal2)
    print("Child 1", child1)
    print("Child 2", child2)
    
    print()
    print()
    graph = [[0,14,22,34,24,28], [14,0,35,70,99,39], [22,35,0,19,91,40], [34,70,19,0,53,50], [24,99,91,53,0,55], [28,39,40,50,55,0]]
    keys = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F"}
    tsp_basic = TravellingSalesman(graph, 1, keys)
    savingsTours = tsp_basic.savingsTour(1)
    print("Savings Tour:", savingsTours, "Cost:", tsp_basic.TourCost(savingsTours))