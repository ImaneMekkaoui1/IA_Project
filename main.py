"""
Main script: Runs all experiments and generates figures for the report.
Usage: python main.py
"""


from experiments.graphs import (
    MODULE_GRAPH, MODULE_HEURISTIC_ADMISSIBLE, MODULE_HEURISTIC_GREEDY,
    dict_heuristic, SMALL_GRAPH, SMALL_HEURISTIC
)
from experiments.benchmarks import (
    step_by_step_greedy, step_by_step_astar,
    heuristic_comparison_experiment, weighted_astar_experiment,
    scalability_experiment, plot_nodes_expanded,
    plot_weighted_astar, plot_markov_absorption, plot_heuristic_comparison
)
from markov.absorbing_chain import analyze, print_analysis
from markov.simulation import compare_analytical_vs_simulation


def main():
    print("\n" + "#"*70)
    print("  MINI-PROJET: Recherche Heuristique & MDP")
    print("  ENSET Mohammedia — Master SDIA 2025-2026")
    print("#"*70)

    # ─────────────────────────────────────────────
    # PART II: Step-by-step traces on module graph
    # ─────────────────────────────────────────────
    print("\n\n" + "─"*60)
    print("PART II — Analytical Case Study")
    print("─"*60)

    START, GOAL = 'S', 'G'

    step_by_step_greedy(MODULE_GRAPH, START, GOAL, MODULE_HEURISTIC_GREEDY)
    step_by_step_astar(MODULE_GRAPH, START, GOAL, MODULE_HEURISTIC_ADMISSIBLE)

    # ─────────────────────────────────────────────
    # PART II.5: Heuristic comparison
    # ─────────────────────────────────────────────
    rows = heuristic_comparison_experiment(MODULE_GRAPH, START, GOAL)
    plot_heuristic_comparison(rows, "experiments/heuristic_comparison.png")

    # ─────────────────────────────────────────────
    # PART IV: Scalability experiment
    # ─────────────────────────────────────────────
    print("\n\n" + "─"*60)
    print("PART IV — Scalability Experiment")
    print("─"*60)
    sizes, algo_results = scalability_experiment()
    plot_nodes_expanded(sizes, algo_results, "experiments/nodes_expanded.png")

    
    wa_results, opt_cost = weighted_astar_experiment(
        MODULE_GRAPH, START, GOAL, MODULE_HEURISTIC_ADMISSIBLE
    )
    plot_weighted_astar(wa_results, opt_cost, "experiments/weighted_astar.png")

    # ─────────────────────────────────────────────
    # PART V: Markov chain analysis
    # ─────────────────────────────────────────────
    print("\n\n" + "─"*60)
    print("PART V — Absorbing Markov Chain")
    print("─"*60)

    
    results_sym = analyze(p=0.5)
    print_analysis(results_sym)
    compare_analytical_vs_simulation(results_sym, n_trials=200000)
    plot_markov_absorption(results_sym, n_trials=200000,
                           save_path="experiments/markov_absorption.png")

    
    print("\n─ Extension E4: Asymmetric case (p = 0.6) ─")
    results_asym = analyze(p=0.6)
    print_analysis(results_asym)
    compare_analytical_vs_simulation(results_asym, n_trials=100000)
    plot_markov_absorption(results_asym, n_trials=100000,
                           save_path="experiments/markov_absorption_asym.png")

    print("\n\n✓ All experiments complete. Figures saved in experiments/")

main()