"""
Graph definitions and automatic graph generator for experiments.
Includes the reference graph from the module and additional test graphs.
"""
import random
import math


# ──────────────────────────────────────────────
#  Reference graph from the course module (Pr. MESTARI)
#  Nodes: S, A, B, C, D, G  (S = start, G = goal)
#  Edges: S->A(1), S->B(4), A->C(2), A->D(5), B->D(1), C->G(5), D->G(3)
#  h(S)=7, h(A)=6, h(B)=5, h(C)=4, h(D)=2, h(G)=0
# ──────────────────────────────────────────────
MODULE_GRAPH = {
    'S': [('A', 1), ('B', 4)],
    'A': [('C', 2), ('D', 5)],
    'B': [('D', 1)],
    'C': [('G', 5)],
    'D': [('G', 3)],
    'G': [],
}

MODULE_START = 'S'
MODULE_GOAL  = 'G'

# Heuristic values h(n, G) — admissible and coherent (from course support)
MODULE_HEURISTIC_ADMISSIBLE = {
    'S': 7, 'A': 6, 'B': 5, 'C': 4, 'D': 2, 'G': 0
}

# Non-coherent heuristic (admissible but violates triangle inequality)
# Violates: h(S)=7 > c(S,A)+h(A) = 1+6 = 7  -> tweak D to break consistency
MODULE_HEURISTIC_NON_COHERENT = {
    'S': 7, 'A': 6, 'B': 5, 'C': 4, 'D': 3, 'G': 0
}

# Non-admissible heuristic (overestimates)
MODULE_HEURISTIC_NON_ADMISSIBLE = {
    'S': 10, 'A': 9, 'B': 8, 'C': 7, 'D': 6, 'G': 0
}

# Greedy heuristic (used for Greedy BFS step-by-step demo)
MODULE_HEURISTIC_GREEDY = MODULE_HEURISTIC_ADMISSIBLE


def dict_heuristic(h_dict):
    """Wrap a heuristic dict as a callable h(node, goal)."""
    return lambda node, goal: h_dict.get(node, 0)


# ──────────────────────────────────────────────
#  Small graph for quick validation
# ──────────────────────────────────────────────
SMALL_GRAPH = {
    'S': [('A', 3), ('B', 1)],
    'A': [('C', 3), ('B', 2)],
    'B': [('C', 4), ('D', 1)],
    'C': [('G', 2)],
    'D': [('G', 5)],
    'G': [],
}

SMALL_HEURISTIC = {
    'S': 7, 'A': 5, 'B': 6, 'C': 2, 'D': 3, 'G': 0
}


# ──────────────────────────────────────────────
#  Automatic graph generator (Extension E2)
# ──────────────────────────────────────────────
def generate_random_graph(n_nodes, n_edges, max_cost=10, seed=None):
    """
    Generate a random directed weighted graph.

    Args:
        n_nodes: number of nodes
        n_edges: number of directed edges
        max_cost: maximum edge weight
        seed: random seed for reproducibility

    Returns:
        (graph dict, start, goal, heuristic callable)
    """
    if seed is not None:
        random.seed(seed)

    nodes = list(range(n_nodes))
    graph = {n: [] for n in nodes}

    
    for i in range(n_nodes - 1):
        cost = random.randint(1, max_cost)
        graph[i].append((i + 1, cost))

    added = 0
    attempts = 0
    while added < n_edges - (n_nodes - 1) and attempts < 10000:
        u = random.randint(0, n_nodes - 2)
        v = random.randint(u + 1, n_nodes - 1)
        cost = random.randint(1, max_cost)
        if (v, cost) not in graph[u]:
            graph[u].append((v, cost))
            added += 1
        attempts += 1

    start = 0
    goal = n_nodes - 1

    
    def h(node, g):
        return max(0, (g - node) * 0.5)

    return graph, start, goal, h


def generate_grid_graph(rows, cols, max_cost=5, seed=None):
    """
    Generate a grid graph with random edge weights.
    Nodes are (row, col) tuples.
    Edges: right and down only (to keep it a DAG).
    """
    if seed is not None:
        random.seed(seed)

    graph = {}
    for r in range(rows):
        for c in range(cols):
            node = (r, c)
            graph[node] = []
            if c + 1 < cols:
                graph[node].append(((r, c + 1), random.randint(1, max_cost)))
            if r + 1 < rows:
                graph[node].append(((r + 1, c), random.randint(1, max_cost)))

    start = (0, 0)
    goal = (rows - 1, cols - 1)

    def h_euclidean(node, g):
        dr = g[0] - node[0]
        dc = g[1] - node[1]
        return math.sqrt(dr * dr + dc * dc)

    def h_manhattan(node, g):
        return abs(g[0] - node[0]) + abs(g[1] - node[1])

    return graph, start, goal, h_manhattan