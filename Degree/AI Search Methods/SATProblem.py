from HeuristicSearch import BeamSearch

class SATProblem:
    def __init__(self, clauses):
        self.clauses = clauses

    def Heuristic(self, assignment):
        # Will determine the number of clauses that are satisfied by the assignment
        satisfied = 0
        for clause in self.clauses:
            for literal in clause:
                if (literal > 0 and assignment[literal - 1] == 1) or (literal < 0 and assignment[-literal - 1] == 0):
                    satisfied += 1
                    break
        return satisfied
    
    def GoalTest(self, assignment):
        # Will return True if CNF is satisfied by the assignment
        satisfied = self.Heuristic(assignment)
        if satisfied == len(self.clauses): return True
        return False
    
    def MoveGen(self, assignment):
        # This will return new assignments that can be reached from the current assignment
        moves = []
        for i in range(len(assignment)):
            newState = assignment.copy()
            newState[i] = 1 - newState[i]
            moves.append(newState)
        return moves

def main():
    # F(a, b, c, d, e) = (a ∨ ¬b) ∧ (¬a ∨ ¬c) ∧ (¬a ∨ ¬e) ∧ (b ∨ ¬e) ∧ (¬c ∨ d) ∧ (c ∨ e)
    clauses = [[1, -2], [-1, -3], [-1, -5], [2, -5], [-3, 4], [3, 5]]
    S = [1,0,1,0,1]
    A = [0,0,1,0,1]
    B = [1,1,1,0,1]
    C = [1,0,0,0,1]
    D = [1,0,1,1,1]
    E = [1,0,1,0,0]
    moves = [S, A, B, C, D, E]
    SAT = SATProblem(clauses)
    print("Order is: S, A, B, C, D, E")
    print("Heuristic:")
    for move in moves:
        print(SAT.Heuristic(move), end=" ")
    print()
    moves = SAT.MoveGen([0, 0, 1, 0, 1])
    for move in moves:
        print(move, SAT.Heuristic(move))
    print()
    bestNode = BeamSearch(S, SAT.MoveGen, SAT.GoalTest, SAT.Heuristic, 2, "max")
    print("Best Node:", bestNode)

if __name__ == "__main__":
    main()