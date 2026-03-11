"""
Benchmarks and experimental comparison of search algorithms.
Generates tables and plots for the report.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from search.astar import astar, weighted_astar
from search.ucs import ucs
from search.greedy import greedy_bfs
from search.utils import SearchLogger
from experiments.graphs import (
    MODULE_GRAPH, MODULE_HEURISTIC_ADMISSIBLE,
    MODULE_HEURISTIC_NON_COHERENT, MODULE_HEURISTIC_NON_ADMISSIBLE,
    MODULE_HEURISTIC_GREEDY, dict_heuristic,
    generate_random_graph, generate_grid_graph, SMALL_GRAPH, SMALL_HEURISTIC
)


def run_all_algorithms(graph, start, goal, heuristic_dict):
    """Run all algorithms on a graph and collect stats."""
    h = dict_heuristic(heuristic_dict)
    results = {}

    for name, algo, kwargs in [
        ("UCS",         ucs,           {}),
        ("Greedy",      greedy_bfs,    {"heuristic": h}),
        ("A*",          astar,         {"heuristic": h}),
        ("W-A*(w=1.5)", weighted_astar, {"heuristic": h, "w": 1.5}),
        ("W-A*(w=3.0)", weighted_astar, {"heuristic": h, "w": 3.0}),
    ]:
        log = SearchLogger(name)
        if name == "UCS":
            path, cost = algo(graph, start, goal, logger=log)
        else:
            path, cost = algo(graph, start, goal, logger=log, **kwargs)
        results[name] = log.report()
        results[name]["path"] = path
        results[name]["cost"] = cost

    return results


def step_by_step_greedy(graph, start, goal, heuristic_dict):
    """Print detailed Greedy BFS trace."""
    import heapq
    from search.utils import reconstruct_path

    h = dict_heuristic(heuristic_dict)
    print("\n" + "="*60)
    print("GREEDY BEST-FIRST SEARCH — Step-by-step trace")
    print(f"Start: {start}, Goal: {goal}")
    print("="*60)

    counter = 0
    open_list = [(h(start, goal), counter, start)]
    counter += 1
    came_from = {start: None}
    g_score = {start: 0.0}
    open_set = {start}
    closed_set = set()
    step = 0

    while open_list:
        h_val, _, current = heapq.heappop(open_list)
        if current not in open_set:
            continue
        open_set.discard(current)
        step += 1

        frontier_display = [(n, round(h(n, goal),1)) for n in open_set]
        print(f"\nStep {step}: Expand '{current}' [h={h_val}]")
        print(f"  Frontier (node, h): {frontier_display}")

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            cost = sum(
                next(c for n, c in graph.get(path[i], []) if n == path[i+1])
                for i in range(len(path)-1)
            )
            print(f"\n  GOAL REACHED!")
            print(f"  Path: {' -> '.join(map(str, path))}")
            print(f"  Cost: {cost}")
            return path, cost

        closed_set.add(current)
        for neighbor, edge_cost in graph.get(current, []):
            if neighbor in closed_set or neighbor in open_set:
                continue
            came_from[neighbor] = current
            g_score[neighbor] = g_score[current] + edge_cost
            heapq.heappush(open_list, (h(neighbor, goal), counter, neighbor))
            counter += 1
            open_set.add(neighbor)
            print(f"  Added '{neighbor}' to frontier [h={h(neighbor,goal)}]")

    print("  No path found.")
    return None, float('inf')


def step_by_step_astar(graph, start, goal, heuristic_dict):
    """Print detailed A* trace."""
    import heapq
    from search.utils import reconstruct_path

    h = dict_heuristic(heuristic_dict)
    print("\n" + "="*60)
    print("A* SEARCH — Step-by-step trace")
    print(f"Start: {start}, Goal: {goal}")
    print("="*60)

    counter = 0
    open_list = [(h(start, goal), counter, start)]
    counter += 1
    came_from = {start: None}
    g_score = {start: 0.0}
    open_set = {start}
    closed_set = set()
    step = 0

    while open_list:
        f_val, _, current = heapq.heappop(open_list)
        if current not in open_set:
            continue
        open_set.discard(current)
        step += 1

        g_val = g_score[current]
        h_val = h(current, goal)
        frontier_display = [(n, round(g_score.get(n,0)+h(n,goal),1)) for n in open_set]
        print(f"\nStep {step}: Expand '{current}' [g={g_val}, h={h_val}, f={f_val:.1f}]")
        print(f"  Frontier (node, f): {frontier_display}")

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            cost = g_score[goal]
            print(f"\n  GOAL REACHED!")
            print(f"  Path: {' -> '.join(map(str, path))}")
            print(f"  Optimal cost: {cost}")
            return path, cost

        closed_set.add(current)
        for neighbor, edge_cost in graph.get(current, []):
            if neighbor in closed_set:
                continue
            tentative_g = g_score[current] + edge_cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + h(neighbor, goal)
                heapq.heappush(open_list, (f, counter, neighbor))
                counter += 1
                open_set.add(neighbor)
                print(f"  Updated '{neighbor}': g={tentative_g}, h={h(neighbor,goal)}, f={f:.1f}")

    print("  No path found.")
    return None, float('inf')


def heuristic_comparison_experiment(graph, start, goal):
    """Compare admissible, non-coherent, non-admissible heuristics."""
    print("\n" + "="*60)
    print("HEURISTIC COMPARISON EXPERIMENT")
    print("="*60)

    heuristics = [
        ("Admissible + Coherent", MODULE_HEURISTIC_ADMISSIBLE, True),
        ("Admissible, Non-Coherent", MODULE_HEURISTIC_NON_COHERENT, False),
        ("Non-Admissible", MODULE_HEURISTIC_NON_ADMISSIBLE, True),
    ]

    rows = []
    for h_name, h_dict, coherent in heuristics:
        h = dict_heuristic(h_dict)
        log = SearchLogger("A*")
        path, cost = astar(graph, start, goal, h, coherent=coherent, logger=log)
        r = log.report()
        rows.append({
            "Heuristic": h_name,
            "Nodes Expanded": r["nodes_expanded"],
            "Path": " -> ".join(map(str, path)) if path else "None",
            "Cost": cost,
        })
        print(f"\n{h_name}:")
        print(f"  Nodes expanded : {r['nodes_expanded']}")
        print(f"  Path           : {' -> '.join(map(str, path)) if path else 'None'}")
        print(f"  Cost           : {cost}")

    return rows


def weighted_astar_experiment(graph, start, goal, h_dict):
    """Extension E3: Compare exact vs Weighted A* for several w values."""
    print("\n" + "="*60)
    print("WEIGHTED A* EXPERIMENT (Extension E3)")
    print("="*60)

    h = dict_heuristic(h_dict)
    log0 = SearchLogger("A*(w=1)")
    opt_path, opt_cost = astar(graph, start, goal, h, logger=log0)
    opt_nodes = log0.report()["nodes_expanded"]
    print(f"\nOptimal (A*, w=1): cost={opt_cost}, nodes={opt_nodes}")

    results = []
    for w in [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
        log = SearchLogger(f"W-A*(w={w})")
        path, cost = weighted_astar(graph, start, goal, h, w=w, logger=log)
        r = log.report()
        subopt_ratio = cost / opt_cost if opt_cost else None
        results.append({
            "w": w, "cost": cost, "nodes": r["nodes_expanded"],
            "ratio": subopt_ratio
        })
        print(f"  w={w:.1f}: cost={cost}, nodes={r['nodes_expanded']}, "
              f"ratio={subopt_ratio:.3f}" if subopt_ratio else "")

    return results, opt_cost


def plot_nodes_expanded(graph_sizes, algo_results, save_path="experiments/nodes_expanded.png"):
    """Plot nodes expanded vs graph size for each algorithm."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"UCS": "#E74C3C", "Greedy": "#3498DB", "A*": "#2ECC71",
              "W-A*(w=1.5)": "#F39C12", "W-A*(w=3.0)": "#9B59B6"}

    for algo in ["UCS", "Greedy", "A*", "W-A*(w=1.5)", "W-A*(w=3.0)"]:
        nodes = [r[algo]["nodes_expanded"] for r in algo_results]
        ax.plot(graph_sizes, nodes, marker='o', label=algo, color=colors[algo], linewidth=2)

    ax.set_xlabel("Graph Size (nodes)", fontsize=12)
    ax.set_ylabel("Nodes Expanded", fontsize=12)
    ax.set_title("Nodes Expanded vs Graph Size", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_weighted_astar(results, opt_cost, save_path="experiments/weighted_astar.png"):
    """Plot cost ratio and nodes vs w for Weighted A*."""
    ws = [r["w"] for r in results]
    ratios = [r["ratio"] for r in results]
    nodes = [r["nodes"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(ws, ratios, 'o-', color='#E74C3C', linewidth=2, markersize=7)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Optimal')
    ax1.set_xlabel("w", fontsize=12)
    ax1.set_ylabel("Cost / Optimal Cost", fontsize=12)
    ax1.set_title("Solution Quality vs w", fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ws, nodes, 's-', color='#3498DB', linewidth=2, markersize=7)
    ax2.set_xlabel("w", fontsize=12)
    ax2.set_ylabel("Nodes Expanded", fontsize=12)
    ax2.set_title("Search Effort vs w", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_markov_absorption(analytical_results, n_trials=100000,
                           save_path="experiments/markov_absorption.png"):
    """Plot absorption probabilities and expected time."""
    from markov.simulation import monte_carlo

    states = analytical_results['transient_states']
    B = analytical_results['B']
    t = analytical_results['t']

    mc_data = [monte_carlo(s, p=analytical_results['p'], n_trials=n_trials) for s in states]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(states))
    width = 0.35

    # Absorption prob at state 6 (win)
    analyt_win = B[:, 1]
    mc_win = [d['p_win_empirical'] for d in mc_data]

    axes[0].bar(x - width/2, analyt_win, width, label='Analytical', color='#2ECC71', alpha=0.85)
    axes[0].bar(x + width/2, mc_win, width, label='Monte Carlo', color='#3498DB', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'State {s}' for s in states])
    axes[0].set_ylabel("P(Win | start state)", fontsize=11)
    axes[0].set_title("Absorption Probability at State 6", fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Expected time
    mc_time = [d['mean_time_empirical'] for d in mc_data]
    axes[1].plot(states, t, 'o-', color='#E74C3C', linewidth=2, markersize=8, label='Analytical')
    axes[1].plot(states, mc_time, 's--', color='#F39C12', linewidth=2, markersize=8, label='Monte Carlo')
    axes[1].set_xlabel("Initial State", fontsize=11)
    axes[1].set_ylabel("Expected Steps", fontsize=11)
    axes[1].set_title("Expected Absorption Time", fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_heuristic_comparison(rows, save_path="experiments/heuristic_comparison.png"):
    """Bar chart comparing nodes expanded across heuristic types."""
    names = [r["Heuristic"] for r in rows]
    nodes = [r["Nodes Expanded"] for r in rows]
    costs = [r["Cost"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2ECC71', '#F39C12', '#E74C3C']
    bars = ax.bar(names, nodes, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)

    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'cost={cost}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel("Nodes Expanded", fontsize=12)
    ax.set_title("A* Performance by Heuristic Type", fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def scalability_experiment():
    """Run all algorithms on graphs of increasing size and collect stats."""
    sizes = [10, 20, 50, 100, 200, 500]
    algo_results = []

    for n in sizes:
        graph, start, goal, h_fn = generate_random_graph(n, n*2, seed=n)
        h_dict = {node: max(0, (goal - node) * 0.5) for node in range(n)}
        res = run_all_algorithms(graph, start, goal, h_dict)
        algo_results.append(res)
        print(f"  n={n}: UCS={res['UCS']['nodes_expanded']}, "
              f"A*={res['A*']['nodes_expanded']}, "
              f"Greedy={res['Greedy']['nodes_expanded']}")

    return sizes, algo_results