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
    prime = computePrime(Loops, Terminate, Detour, finalVertices)
    return prime, Loops, Terminate, Detour

def computePrime(Loops, Terminate, Detour, finalVertices):
    complete = []
    # Isolate paths that start from the initial vertice and end in a final vertice
    for path in Terminate:
        if path[0] == 0 and path[-1] in finalVertices:
            complete.append(path)
    # For each path, check if there is a loop or detour that can be inserted
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
        additive = 0
        for i, loop in store_detour:
            suffix = complete[j][i+2+additive:]
            complete[j] = complete[j][:i+additive]
            complete[j].extend(loop)
            complete[j].extend(suffix)
            additive += len(loop) - 2
        for i, loop in store:
            suffix = complete[j][i+1+additive:]
            complete[j] = complete[j][:i+additive]
            complete[j].extend(loop)
            complete[j].extend(suffix)
            additive += len(loop) - 1
    return complete

G = [[1,4], [2,5], [3], [1], [4,6], [6], []]
Prime, Loops, Terminate, Detour = computingPrimePath(G, [6])
print("Prime Paths:")
for path in Prime:
    print(path)
print("Loops:")
for path in Loops:
    print(path)
print("Detour:")
for path in Detour:
    print(path)
print("Terminate:")
for path in Terminate:
    print(path)
