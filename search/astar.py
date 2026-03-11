"""
A* Search Algorithm implementation.
Supports coherent and non-coherent heuristics, with CLOSED-list reopening.
"""
import heapq
from search.utils import reconstruct_path, SearchLogger


def astar(graph, start, goal, heuristic, coherent=True, logger=None):
    """
    A* search algorithm.

    Args:
        graph: dict {node: [(neighbor, cost), ...]}
        start: start node
        goal: goal node
        heuristic: callable h(node, goal) -> float
        coherent: if False, allow reopening nodes from CLOSED
        logger: optional SearchLogger instance

    Returns:
        (path, cost) or (None, inf) if no path found
    """
    if logger:
        logger.start()

   
    counter = 0
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), counter, start))
    counter += 1

    came_from = {start: None}
    g_score = {start: 0.0}

    
    open_set = {start}
    closed_set = set()

    while open_list:
        f_current, _, current = heapq.heappop(open_list)

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

            
            if coherent and neighbor in closed_set and tentative_g >= g_score.get(neighbor, float('inf')):
                continue

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)

                
                if not coherent and neighbor in closed_set:
                    closed_set.discard(neighbor)

                heapq.heappush(open_list, (f, counter, neighbor))
                counter += 1
                open_set.add(neighbor)

    if logger:
        logger.log_result(None, float('inf'))
        logger.stop()
    return None, float('inf')


def weighted_astar(graph, start, goal, heuristic, w=1.5, logger=None):
    """
    Weighted A* with f(n) = g(n) + w * h(n).
    w >= 1: w=1 gives standard A*, larger w trades optimality for speed.
    """
    if logger:
        logger.start()

    counter = 0
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal) * w, counter, start))
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
                f = tentative_g + w * heuristic(neighbor, goal)
                heapq.heappush(open_list, (f, counter, neighbor))
                counter += 1
                open_set.add(neighbor)

    if logger:
        logger.log_result(None, float('inf'))
        logger.stop()
    return None, float('inf')