class TravellingSalesman:
    def __init__(self, graph, start, keys):
        self.graph = graph
        self.start = start
        self.keys = keys
        self.reverse = {v:k for k,v in keys.items()}
    
    def isGraphSymmetric(self):
        # Check if the graph is symmetric
        for i in range(len(self.graph)):
            for j in range(len(self.graph[i])):
                if self.graph[i][j] != self.graph[j][i]:
                    return False
        return True
    
    def NearestNeighbor(self, state, visited = []):
        # return the nearest neighbor of the current state
        # State is city
        neighbors = self.graph[state]
        nearest = max(neighbors)
        index = -1
        for i in range(len(neighbors)):
            if neighbors[i] < nearest and i != state and i not in visited:
                nearest = neighbors[i]
                index = i
        
        return index
    
    def computeTour(self):
        # Compute the tour using nearest neighbor algorithm
        tour = [self.start]
        current = self.start
        while len(tour) < len(self.graph):
            current = self.NearestNeighbor(current, tour)
            tour.append(current)
        tour.append(self.start)
        return [self.keys[i] for i in tour]
    
    def GreedyHeuristic(self, tour = []):
        # Return the shortest edge from the graph as long as the edge doesn't complete a cycle
        search = []
        if tour == []: search = [i for i in range(len(self.graph))]
        else: 
            search = [tour[0], tour[-1]]
        shortest = max(max(self.graph))
        edge = []
        for i in search:
            for j in range(len(self.graph[i])):
                if self.graph[i][j] < shortest and i != j and j not in tour:
                    shortest = self.graph[i][j]
                    edge = [i, j]
        return edge
        
    def computeGreedyTour(self):
        # Compute the tour using greedy algorithm
        tour = []
        firstEdge = self.GreedyHeuristic()
        tour.append(firstEdge[0])
        tour.append(firstEdge[1])
        while len(tour) < len(self.graph):
            edge = self.GreedyHeuristic(tour)
            if edge[0] == tour[-1]: tour.append(edge[1])
            else: tour.insert(0, edge[1])
        tour.append(tour[0])
        return [self.keys[i] for i in tour]
    
    def TourCost(self, tour):
        # Compute the cost of the tour
        tour = [self.reverse[i] for i in tour]
        cost = 0
        for i in range(len(tour)-1):
            cost += self.graph[tour[i]][tour[i+1]]
        return cost
    
    def twoCityExchange(self, prevTour, city1, city2):
        # Exchange two cities in the tour
        tour = prevTour.copy()
        i = tour.index(city1)
        j = tour.index(city2)
        tour[i], tour[j] = tour[j], tour[i]
        if city1 == tour[0]: tour[-1] = city1
        if city2 == tour[0]: tour[-1] = city2
        return tour
    
    def twoEdgeExchange(self, prevTour, city1, city2, city3, city4):
        # Exchange two edges in the tour
        tour = prevTour.copy()
        i = tour.index(city1)
        j = tour.index(city2)
        k = tour.index(city3)
        l = tour.index(city4)
        tour[j], tour[k] = tour[k], tour[j]
        if city1 == tour[0]: tour[-1] = city1
        if city2 == tour[0]: tour[-1] = city2
        if city3 == tour[0]: tour[-1] = city3
        if city4 == tour[0]: tour[-1] = city4
        return tour
    
    def subtours(self, start):
        # Find all subtours starting from the given city
        subtours = []
        for i in range(len(self.graph)):
            if self.graph[start][i] != 0:
                subtours.append([self.keys[start], self.keys[i], self.keys[start]])
        return subtours
    
    def computeSavings(self, depot):
        # Compute the savings for each pair of cities
        savings = []
        for i in range(len(self.graph)):
            for j in range(i+1, len(self.graph[i])):
                if i == j or i == 1 or j == 1: continue
                savings.append((self.graph[depot][i] + self.graph[depot][j] - self.graph[i][j], self.keys[i], self.keys[j]))
        return savings
    
    def savingsTour(self, depot):
        savings = self.computeSavings(depot)
        # Sorting the savings
        savings.sort(key=lambda x: x[0], reverse=True)
        depot = self.keys[depot]
        tour = [depot, savings[0][1], savings[0][2], depot]
        for i in range(1, len(savings)):
            saving = savings[i]
            if saving[1] in tour and saving[2] in tour: continue
            if saving[1] in tour:
                index = tour.index(saving[1])
                if tour[index-1] == depot: tour.insert(index, saving[2])
                elif tour[index+1] == depot: tour.insert(index+1, saving[2])
            elif saving[2] in tour:
                index = tour.index(saving[2])
                if tour[index-1] == depot: tour.insert(index, saving[1])
                elif tour[index+1] == depot: tour.insert(index+1, saving[1])
        return tour
            

def TSPtest():
    keys = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F", 6:"G"}
    graph = [[0, 50, 36, 28, 30, 72, 50], [50, 0, 82, 36, 58, 41, 71], [36, 82, 0, 50, 32, 92, 42], [28, 36, 50, 0, 22, 45, 36],
             [30, 58, 32, 22, 0, 61, 20], [72, 41, 92, 45, 61, 0, 61], [50, 71, 42, 36, 20, 61, 0]]
    TSP = TravellingSalesman(graph, 1, keys)
    print("Is graph symmetric:", TSP.isGraphSymmetric())
    nnTour = TSP.computeTour()
    print("Nearest Neighbor Tour:", nnTour, "Cost:", TSP.TourCost(nnTour))
    greedyTour = TSP.computeGreedyTour()
    print("Greedy Tour:", greedyTour, "Cost:", TSP.TourCost(greedyTour))
    twoCityExchange = TSP.twoCityExchange(greedyTour, "B", "E")
    print("Two City Exchange:", twoCityExchange, "Cost:", TSP.TourCost(twoCityExchange))
    twoEdgeExchange = TSP.twoEdgeExchange(greedyTour, "B", "C", "D", "E")
    print("Two Edge Exchange:", twoEdgeExchange, "Cost:", TSP.TourCost(twoEdgeExchange))
    subtours = TSP.subtours(1)
    for subtour in subtours:
        print("Subtour:", subtour, "Cost:", TSP.TourCost(subtour))
    savings = TSP.computeSavings(1)
    for saving in savings:
        print("Savings:", saving)
    # Sorting the savings
    savings.sort(key=lambda x: x[0], reverse=True)
    print("Sorted Savings:")
    for saving in savings:
        print("Savings:", saving)
    savingsTours = TSP.savingsTour(1)
    print("Savings Tour:", savingsTours, "Cost:", TSP.TourCost(savingsTours))
    
def main():
    TSPtest()