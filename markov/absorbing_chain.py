"""
Absorbing Markov Chain Analysis.
Models the Gambler's Ruin problem with states {0,1,2,3,4,5,6},
absorbing states: 0 (ruin) and 6 (win).
"""
import numpy as np


def build_transition_matrix(p=0.5, n_states=7):
    """
    Build transition matrix for Gambler's Ruin.
    State i: win with prob p -> i+1, lose with prob (1-p) -> i-1.
    States 0 and 6 are absorbing.

    Args:
        p: probability of winning each bet
        n_states: total number of states (0..n_states-1)

    Returns:
        P: (n_states x n_states) numpy array
    """
    P = np.zeros((n_states, n_states))
    q = 1 - p

    
    P[0, 0] = 1.0
    P[n_states - 1, n_states - 1] = 1.0

    
    for i in range(1, n_states - 1):
        P[i, i + 1] = p
        P[i, i - 1] = q

    return P


def canonical_form(P, absorbing_states, transient_states):
    """
    Reorder P into canonical form:
    P = [[I, 0], [R, Q]]
    where rows/cols are ordered: absorbing first, then transient.

    Returns:
        Q: transient-to-transient submatrix
        R: transient-to-absorbing submatrix
    """
    all_states = absorbing_states + transient_states
    n_abs = len(absorbing_states)
    n_tr = len(transient_states)

    
    P_canon = P[np.ix_(all_states, all_states)]

    Q = P_canon[n_abs:, n_abs:]  
    R = P_canon[n_abs:, :n_abs]   

    return Q, R, P_canon


def fundamental_matrix(Q):
    """
    Compute fundamental matrix N = (I - Q)^{-1}.
    N[i,j] = expected number of times in transient state j starting from i.
    """
    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)
    return N


def absorption_probabilities(N, R):
    """
    Compute absorption probability matrix B = N * R.
    B[i,j] = probability of being absorbed in absorbing state j starting from transient state i.
    """
    return N @ R


def expected_absorption_time(N):
    """
    Expected number of steps before absorption, starting from each transient state.
    t = N * 1  (ones vector)
    """
    return N @ np.ones(N.shape[1])


def analyze(p=0.5):
    """
    Full analysis of the Gambler's Ruin with given win probability p.
    Returns a dict with all analytical results.
    """
    n = 7
    P = build_transition_matrix(p=p, n_states=n)

    absorbing = [0, 6]
    transient = [1, 2, 3, 4, 5]

    Q, R, P_canon = canonical_form(P, absorbing, transient)
    N = fundamental_matrix(Q)
    B = absorption_probabilities(N, R)
    t = expected_absorption_time(N)

    return {
        "P": P,
        "P_canon": P_canon,
        "Q": Q,
        "R": R,
        "N": N,
        "B": B,
        "t": t,
        "transient_states": transient,
        "absorbing_states": absorbing,
        "p": p,
    }


def print_analysis(results):
    """Pretty-print analysis results."""
    print("\n" + "="*60)
    print(f"GAMBLER'S RUIN ANALYSIS (p = {results['p']})")
    print("="*60)

    print("\nTransition Matrix P:")
    print(np.round(results['P'], 3))

    print("\nSubmatrix Q (transient → transient):")
    print(np.round(results['Q'], 3))

    print("\nSubmatrix R (transient → absorbing):")
    print(np.round(results['R'], 3))

    print("\nFundamental Matrix N = (I-Q)^{-1}:")
    print(np.round(results['N'], 4))

    print("\nAbsorption Probabilities B = N*R:")
    print("  (rows: transient states 1-5, cols: absorbing states 0, 6)")
    for i, s in enumerate(results['transient_states']):
        print(f"  State {s}: P(absorbed at 0) = {results['B'][i,0]:.4f},  "
              f"P(absorbed at 6) = {results['B'][i,1]:.4f}")

    print("\nExpected Time Before Absorption (starting from each transient state):")
    for i, s in enumerate(results['transient_states']):
        print(f"  State {s}: E[T] = {results['t'][i]:.4f} steps")

    
    idx2 = results['transient_states'].index(2)
    print(f"\n--- Starting from State 2 ---")
    print(f"  P(ruin | start=2)  = {results['B'][idx2,0]:.4f}")
    print(f"  P(win  | start=2)  = {results['B'][idx2,1]:.4f}")
    print(f"  E[T | start=2]     = {results['t'][idx2]:.4f} steps")