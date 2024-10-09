def computingPrimePath(G, finalVertices):
    """
    Compute prime paths of a graph G starting from a given vertice start.
    """
    Loops = [ ]
    Terminate = [ ]
    Detour = [ ]
    Q = [ ]
    for v in range(len(G)):
        Q.append([v])
    while Q:
        path = Q.pop(0)
        if path[-1] in finalVertices:
            Terminate.append(path)
            continue
        for v in G[path[-1]]:
            newPath = path.copy()
            newPath.append(v)
            if v in finalVertices:
                # If the path ends in a final vertice, add it to the terminate list
                Terminate.append(newPath)
            elif v in path and v == path[0]:
                # If the path loops back to the initial vertice, add it to the loops list
                Loops.append(newPath)
            elif v in path:
                # If the path loops back to a vertice that is not the initial vertice, add it to the detour list
                Detour.append(newPath)
            else:
                # Otherwise, add the path to the queue
                Q.append(newPath)
    testpaths = computeTestPaths(Loops, Terminate, Detour, finalVertices)
    return testpaths, Loops, Terminate, Detour

def computeTestPaths(Loops, Terminate, Detour, finalVertices):
    complete = []
    # Isolate paths that start from the initial vertice and end in a final vertice
    for path in Terminate:
        if path[0] == 0 and path[-1] in finalVertices:
            complete.append(path)
    # For each path, check if there is a loop or detour that can be inserted
    testpath = []
    testpath.extend(complete)
    for j, path in enumerate(complete):
        store_detour = []
        store = []
        for i in range(len(path)):
            for detour in Detour:
                if (i != len(path)-1) and (path[i] == detour[0] and path[i+1] == detour[-1]):
                    store_detour.append([i, detour])
            for loop in Loops:
                if path[i] == loop[0]:
                    store.append([i, loop])
        for i, loop in store_detour:
            suffix = complete[j][i+2:]
            newPath = complete[j][:i]
            newPath.extend(loop)
            newPath.extend(suffix)
            testpath.append(newPath)
        for i, loop in store:
            suffix = complete[j][i+1:]
            newPath = complete[j][:i]
            newPath.extend(loop)
            newPath.extend(suffix)
            testpath.append(newPath)
    # Remove duplicates
    testpath = [list(x) for x in set(tuple(x) for x in testpath)]
    return testpath

G = [[1,4], [2,5], [3], [1], [4,6], [6], []]
#G = [[1], [2,5], [3,4], [4], [1], []]
#G = [[1], [2,3], [6], [4,5], [3], [6], []]
TestPaths, Loops, Terminate, Detour = computingPrimePath(G, [6])
print("Test Paths:")
for path in TestPaths:
    print(path)
print("Paths that are Loops:")
for path in Loops:
    print(path)
print("Paths that are a Detour:")
for path in Detour:
    print(path)
print("Paths that Terminate:")
for path in Terminate:
    print(path)
