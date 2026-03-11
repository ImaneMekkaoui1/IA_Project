"""
Greedy Best-First Search implementation.
f(n) = h(n): expands node with minimum heuristic value.
Not guaranteed to find optimal path.
"""
import heapq
from search.utils import reconstruct_path, SearchLogger


def greedy_bfs(graph, start, goal, heuristic, logger=None):
    """
    Greedy Best-First Search.

    Args:
        graph: dict {node: [(neighbor, cost), ...]}
        start: start node
        goal: goal node
        heuristic: callable h(node, goal) -> float
        logger: optional SearchLogger

    Returns:
        (path, cost) or (None, inf)
    """
    if logger:
        logger.start()

    counter = 0
    open_list = [(heuristic(start, goal), counter, start)]
    counter += 1

    came_from = {start: None}
    g_score = {start: 0.0}
    open_set = {start}
    closed_set = set()

    while open_list:
        _, _, current = heapq.heappop(open_list)

        if current not in open_set:
            continue
        open_set.discard(current)

        if logger:
            logger.log_expansion(current, len(open_set))

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
           
            cost = sum(
                next(c for n, c in graph.get(path[i], []) if n == path[i+1])
                for i in range(len(path)-1)
            )
            if logger:
                logger.log_result(path, cost)
                logger.stop()
            return path, cost

        closed_set.add(current)

        for neighbor, edge_cost in graph.get(current, []):
            if neighbor in closed_set:
                continue
            if neighbor not in open_set:
                came_from[neighbor] = current
                g_score[neighbor] = g_score[current] + edge_cost
                h = heuristic(neighbor, goal)
                heapq.heappush(open_list, (h, counter, neighbor))
                counter += 1
                open_set.add(neighbor)

    if logger:
        logger.log_result(None, float('inf'))
        logger.stop()
    return None, float('inf')