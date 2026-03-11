"""
Monte Carlo simulation for Gambler's Ruin validation.
"""
import numpy as np


def simulate_gambler(start_state, p=0.5, n_states=7, max_steps=100000):
    """
    Simulate one run of the Gambler's Ruin.
    Returns (absorbed_state, steps_taken).
    """
    state = start_state
    steps = 0
    absorbing = {0, n_states - 1}

    while state not in absorbing and steps < max_steps:
        if np.random.random() < p:
            state += 1
        else:
            state -= 1
        steps += 1

    return state, steps


def monte_carlo(start_state, p=0.5, n_states=7, n_trials=100000, seed=None):
    """
    Run Monte Carlo simulation for the Gambler's Ruin.

    Returns:
        dict with empirical absorption probabilities and expected time
    """
    if seed is not None:
        np.random.seed(seed)

    absorbed_at = []
    times = []

    for _ in range(n_trials):
        final_state, steps = simulate_gambler(start_state, p, n_states)
        absorbed_at.append(final_state)
        times.append(steps)

    absorbed_at = np.array(absorbed_at)
    times = np.array(times)

    p_ruin = np.mean(absorbed_at == 0)
    p_win = np.mean(absorbed_at == n_states - 1)
    mean_time = np.mean(times)
    std_time = np.std(times)

    return {
        "n_trials": n_trials,
        "start_state": start_state,
        "p": p,
        "p_ruin_empirical": p_ruin,
        "p_win_empirical": p_win,
        "mean_time_empirical": mean_time,
        "std_time": std_time,
    }


def compare_analytical_vs_simulation(analytical_results, n_trials=200000, seed=42):
    """
    Compare analytical results with Monte Carlo simulation.
    Prints comparison table.
    """
    print("\n" + "="*60)
    print(f"MONTE CARLO VALIDATION (n_trials = {n_trials:,})")
    print("="*60)
    print(f"\n{'State':<8} {'P(ruin) analyt':<18} {'P(ruin) MC':<18} "
          f"{'E[T] analyt':<15} {'E[T] MC':<12}")
    print("-"*75)

    if seed is not None:
        np.random.seed(seed)

    for i, s in enumerate(analytical_results['transient_states']):
        mc = monte_carlo(s, p=analytical_results['p'], n_trials=n_trials)
        p_ruin_a = analytical_results['B'][i, 0]
        t_a = analytical_results['t'][i]

        print(f"{s:<8} {p_ruin_a:<18.4f} {mc['p_ruin_empirical']:<18.4f} "
              f"{t_a:<15.2f} {mc['mean_time_empirical']:<12.2f}")

    print()