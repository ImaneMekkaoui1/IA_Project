"""
Uniform Cost Search (UCS) implementation.
f(n) = g(n): expands the node with minimum cumulative cost.
"""
import heapq
from search.utils import reconstruct_path, SearchLogger


def ucs(graph, start, goal, logger=None):
    """
    Uniform Cost Search.

    Args:
        graph: dict {node: [(neighbor, cost), ...]}
        start: start node
        goal: goal node
        logger: optional SearchLogger

    Returns:
        (path, cost) or (None, inf)
    """
    if logger:
        logger.start()

    counter = 0
    open_list = [(0.0, counter, start)]
    counter += 1

    came_from = {start: None}
    g_score = {start: 0.0}
    open_set = {start}
    closed_set = set()

    while open_list:
        g_current, _, current = heapq.heappop(open_list)

        if current not in open_set:
            continue
        open_set.discard(current)

        if logger:
            logger.log_expansion(current, len(open_set))

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            cost = g_score[goal]
            if logger:
                logger.log_result(path, cost)
                logger.stop()
            return path, cost

        closed_set.add(current)

        for neighbor, edge_cost in graph.get(current, []):
            tentative_g = g_score[current] + edge_cost
            if neighbor in closed_set:
                continue
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_list, (tentative_g, counter, neighbor))
                counter += 1
                open_set.add(neighbor)

    if logger:
        logger.log_result(None, float('inf'))
        logger.stop()
    return None, float('inf')