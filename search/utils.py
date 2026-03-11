"""
Utility functions for graph search algorithms.
"""
import time
import heapq
from collections import defaultdict


def reconstruct_path(came_from, start, goal):
    """Reconstruct path from came_from dict."""
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    if path[0] == start:
        return path
    return []


class SearchLogger:
    """Logs search algorithm execution details."""

    def __init__(self, algorithm_name):
        self.algorithm_name = algorithm_name
        self.expansion_order = []
        self.frontier_sizes = []
        self.start_time = None
        self.end_time = None
        self.final_cost = None
        self.final_path = None
        self.iterations = 0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        self.end_time = time.perf_counter()

    def log_expansion(self, node, frontier_size):
        self.expansion_order.append(node)
        self.frontier_sizes.append(frontier_size)
        self.iterations += 1

    def log_result(self, path, cost):
        self.final_path = path
        self.final_cost = cost

    def execution_time(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def report(self):
        return {
            "algorithm": self.algorithm_name,
            "expansion_order": self.expansion_order,
            "nodes_expanded": len(self.expansion_order),
            "frontier_sizes": self.frontier_sizes,
            "final_cost": self.final_cost,
            "final_path": self.final_path,
            "execution_time_ms": self.execution_time() * 1000,
            "iterations": self.iterations,
        }

    def print_report(self):
        r = self.report()
        print(f"\n{'='*50}")
        print(f"Algorithm: {r['algorithm']}")
        print(f"Expansion order: {r['expansion_order']}")
        print(f"Nodes expanded: {r['nodes_expanded']}")
        print(f"Final path: {r['final_path']}")
        print(f"Final cost: {r['final_cost']}")
        print(f"Execution time: {r['execution_time_ms']:.4f} ms")
        print(f"{'='*50}")