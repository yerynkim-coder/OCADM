import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


# =============================
# Problem Generator: SSP Transition Matrix
# =============================
# This function generates random transition probability matrices P(i, j, u)
# for a Stochastic Shortest Path (SSP) problem. The output is a 3D array
# of size (N+1) x (N+1) x M, where:
#   - N: number of non-terminal states
#   - M: number of control inputs
#   - P[i, j, u]: transition probability from state i to state j under input u
#
# Key Properties:
# - Each P[:, :, u] is a stochastic matrix (rows sum to 1).
# - State N is the terminal state: once reached, it stays there.
#   => P[N, :, u] = [0, ..., 0, 1] for all u
# - Transition probabilities are structured: nearby states are more probable.
#   - "Closeness" is based on minimal cyclic distance between states
#   - A scaling matrix is used to assign higher probabilities to neighbors
#   - The SCALE_EXPONENT controls how sharply the probability drops with distance
#
# Parameters:
# - N (int): number of non-terminal states (total states = N+1)
# - M (int): number of control inputs
# - SCALE_EXPONENT (float): controls sharpness of transition preference
#     - 0 → uniform transition probabilities
#     - higher → prefer transitions to nearby states
# - seed (int or None): random seed for reproducibility
#
# Returns:
# - P (np.ndarray): shape (N+1, N+1, M), transition probability matrices
def generate_problem_data(N=10, M=5, SCALE_EXPONENT=2, seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    # First just generate a whole bunch of entries in the range of 0 to 1
    P = np.random.rand(N + 1, N + 1, M)

    # Create scaling matrix, which is used to scale the probabilities so that nearby states, with wrap-around, are more likely
    scale_matrix = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        for j in range(N + 1):
            # gives minimum "distance" between nodes (either go forwards or backwards)
            scale_matrix[i, j] = min((i - j) % N, (j - i) % N)
    
    # scale it: farest away = 1, closest = floor((N+1)/2)+1
    scale_matrix = np.max(scale_matrix) - scale_matrix + 1

    # Accentuate the differences
    scale_matrix **= SCALE_EXPONENT
    
    # Can now scale and normalize the transition probabilities
    for u in range(M):
        # Scale it
        P[:, :, u] *= scale_matrix

        # Normalize it
        for i in range(N):
            P[i, :, u] /= np.sum(P[i, :, u])

        # Set the last row appropriately, corresponding to the termination state
        P[N, :, u] = 0
        P[N, N, u] = 1

    return P

# Plot a typical probability transition matrix under the given control input
def plot_transition_matrix(P, u=0):
    N = P.shape[0] - 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    Z = P[:N, :N, u]
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Node i')
    ax.set_ylabel('Node j')
    ax.set_zlabel('p_ij')
    ax.set_title(f'Transition Probabilities, u = {u + 1}')
    plt.tight_layout()
    plt.show()


# =============================
# Policy Iteration Solver
# =============================
# This class implements the Policy Iteration algorithm for solving a
# Stochastic Shortest Path (SSP) problem.
#
# Inputs:
# - P: Transition probability tensor of shape (n+1, n+1, m),
#      where P[i, j, u] is the probability of transitioning from state i
#      to state j under control input u.
#
# Outputs (from `solve()` method):
# - J: Optimal cost-to-go for each state, shape (n+1,)
# - F: Optimal policy (action index starting from 1), shape (n+1,)
#
# Notes:
# - Assumes stage cost is 1 for non-terminal states, 0 for the terminal state.
# - The terminal state is state `n`, i.e., index `n` in 0-based Python indexing.
# - The method is guaranteed to converge in a finite number of iterations.
class SSPPolicyIteration:
    def __init__(self, P):
        self.P = P
        self.n = P.shape[0] - 1  # number of non-terminal states
        self.m = P.shape[2]      # number of control inputs

        # Stage cost: cost of 1 for all non-terminal states, 0 for terminal state
        self.G = np.ones((self.n + 1, self.m))
        self.G[self.n, :] = 0

    def solve(self):
        """
        Perform policy iteration to compute the optimal cost and policy.

        Returns:
            J (np.ndarray): optimal cost-to-go vector of shape (n+1,)
            F (np.ndarray): optimal policy vector of shape (n+1,) with actions in [1, ..., m]
        """
        F = np.ones(self.n + 1, dtype=int)  # Initial policy: action 1 for all states
        J = np.zeros(self.n + 1)            # Initial cost-to-go
        iter_count = 0

        while True:
            iter_count += 1
            print(f"Policy Iteration Number {iter_count}")

            # === 1) POLICY EVALUATION ===
            J_prev = J.copy()

            A = np.eye(self.n + 1)
            B = np.zeros(self.n + 1)

            # Build system of equations: A * J = B
            for i in range(self.n + 1):
                u = F[i] - 1
                A[i, :] -= self.P[i, :, u]
                B[i] = self.G[i, u]

            # Solve only for first n rows (terminal state's equation is trivial)
            J[:self.n] = np.linalg.solve(A[:self.n, :self.n], B[:self.n])

            # === 2) POLICY IMPROVEMENT ===
            F_prev = F.copy()
            for i in range(self.n):
                # Compute expected cost for each control input at state i
                costs = self.G[i, :] + self.P[i, :, :].T @ J
                F[i] = np.argmin(costs) + 1  # Select action minimizing cost

            # === 3) CONVERGENCE CHECK ===
            if np.all(F == F_prev) or np.linalg.norm(J - J_prev) < 1e-10:
                print("Policy iteration converged.")
                break

        return J, F



# =============================
# Value Iteration Solver
# =============================
# This class implements the Value Iteration algorithm for solving a
# Stochastic Shortest Path (SSP) problem.
#
# Inputs:
# - P: Transition probability tensor of shape (n+1, n+1, m),
#      where P[i, j, u] is the probability of transitioning from state i
#      to state j under control input u.
#
# Outputs (from `solve()` method):
# - J: Optimal cost-to-go for each state, shape (n+1,)
# - F: Optimal policy (action index starting from 1), shape (n+1,)
#
# Notes:
# - Stage cost is 1 for all non-terminal states, and 0 for terminal state.
# - Terminal state is state `n`, i.e., index `n` in 0-based Python indexing.
# - Convergence is determined when value function changes are below `epsilon`.
class SSPValueIteration:
    def __init__(self, P):
        self.P = P
        self.n = P.shape[0] - 1  # number of non-terminal states
        self.m = P.shape[2]      # number of control inputs

        # Stage cost: 1 per step, except for terminal state which has zero cost
        self.G = np.ones((self.n + 1, self.m))
        self.G[self.n, :] = 0

    def solve(self, epsilon=0.1):
        """
        Perform value iteration to compute the optimal cost and policy.

        Parameters:
            epsilon (float): convergence tolerance for stopping condition

        Returns:
            J (np.ndarray): optimal cost-to-go vector of shape (n+1,)
            F (np.ndarray): optimal policy vector of shape (n+1,) with actions in [1, ..., m]
        """
        # === 0) INITIALIZATION ===
        J = np.zeros(self.n + 1)            # initial cost-to-go
        F = np.ones(self.n + 1, dtype=int)  # initial policy: action 1 for all states
        iter_count = 0

        while True:
            iter_count += 1
            print(f"Value Iteration Number {iter_count}")
            J_prev = J.copy()

            # === 1) VALUE FUNCTION UPDATE ===
            for i in range(self.n):
                # Compute expected cost for each control input at state i
                costs = self.G[i, :] + self.P[i, :, :].T @ J
                J[i] = np.min(costs)
                F[i] = np.argmin(costs) + 1

            # === 2) CONVERGENCE CHECK ===
            delta_J = np.linalg.norm(J - J_prev)
            if delta_J < epsilon:
                print(f"Convergence criteria reached (delta J = {delta_J:.2f}).")
                break

        return J, F



# =============================
# Linear Programming Solver
# =============================
# This class solves a Stochastic Shortest Path (SSP) problem using Linear Programming.
#
# Inputs:
# - P: Transition probability tensor of shape (n+1, n+1, m),
#      where P[i, j, u] is the probability of transitioning from state i
#      to state j under control input u.
#
# Outputs (from `solve()` method):
# - J: Optimal cost-to-go for each state, shape (n+1,)
# - F: Optimal policy (action index starting from 1), shape (n+1,)
#
# Notes:
# - Stage cost is 1 for all non-terminal states and 0 for the terminal state.
# - Only the first `n` cost values are decision variables (excluding terminal state).
# - The policy is derived after solving the LP, by evaluating actions using the cost.
class SSPLinearProgram:
    def __init__(self, P):
        self.P = P
        self.n = P.shape[0] - 1  # number of non-terminal states
        self.m = P.shape[2]      # number of control inputs

        # Stage cost: 1 per step, 0 at terminal state
        self.G = np.ones((self.n + 1, self.m))
        self.G[self.n, :] = 0

    def solve(self):
        """
        Solve the SSP problem using linear programming.

        Returns:
            J (np.ndarray): optimal cost-to-go vector of shape (n+1,)
            F (np.ndarray): optimal policy vector of shape (n+1,) with actions in [1, ..., m]
        """
        # === 1) SETUP LINEAR PROGRAM ===
        # We solve a large LP:
        #   Maximize sum(x) (which becomes minimize -sum(x))
        #   Subject to x[i] >= expected cost from each possible action
        #   for each state i and control input u

        A = []
        b = []

        for i in range(self.n):
            Pi = self.P[i, :self.n, :]  # exclude terminal state's cost-to-go
            for u in range(self.m):
                row = np.zeros(self.n)
                row[i] = 1                      # cost-to-go at state i
                row -= Pi[:, u]                # subtract expected next cost
                A.append(row)
                b.append(self.G[i, u])         # stage cost for input u at state i

        A = np.array(A)  # Shape: (n * m, n)
        b = np.array(b)

        # Objective: maximize sum of x => minimize -sum(x)
        f = -np.ones(self.n)

        # Solve using scipy.optimize.linprog
        res = linprog(f, A_ub=A, b_ub=b, method='highs')

        # === 2) DERIVE COST AND POLICY ===
        J = np.zeros(self.n + 1)
        J[:self.n] = res.x                   # optimal cost-to-go
        F = np.ones(self.n + 1, dtype=int)   # policy initialized to action 1

        # Recover policy by finding the action that minimizes cost-to-go
        for i in range(self.n):
            costs = self.G[i, :] + self.P[i, :, :].T @ J
            F[i] = np.argmin(costs) + 1

        return J, F



# =============================
# Simulation: Stochastic Shortest Path
# =============================
# Simulates multiple realizations of an SSP process under a given policy,
# and returns the total cost incurred in each simulation.
#
# Inputs:
# - P: Transition probability tensor of shape (n+1, n+1, m),
#      where P[i, j, u] is the probability of transitioning from state i
#      to state j under control input u.
# - i0: Initial state (starting node), integer in [0, n].
# - SIM_NUM: Number of simulations to run.
# - policy: Optimal policy vector of shape (n+1,), where each entry is an
#      integer in [1, m] representing the chosen control input for each state.
#
# Outputs:
# - sim_costs: Array of shape (SIM_NUM,), containing total cost for each simulation.
def simulate_ssp(P, i0, SIM_NUM, policy):
    """
    Simulate the SSP system multiple times using the given policy.

    Parameters:
        P (np.ndarray): transition probability matrix of shape (n+1, n+1, m)
        i0 (int): starting node
        SIM_NUM (int): number of simulations
        policy (np.ndarray): optimal policy (1-based indexing), shape (n+1,)

    Returns:
        sim_costs (np.ndarray): simulated cost from each run, shape (SIM_NUM,)
    """
    n = P.shape[0] - 1  # number of non-terminal states
    m = P.shape[2]      # number of control inputs

    # Stage costs: 1 for all non-terminal states, 0 for terminal state
    G = np.ones((n + 1, m))
    G[n, :] = 0

    # Precompute cumulative transition matrices for faster sampling
    PP = np.zeros_like(P)
    for u in range(m):
        PP[:, :, u] = np.cumsum(P[:, :, u], axis=1)

    # Run multiple simulations from starting state i0
    sim_costs = np.zeros(SIM_NUM)
    for s in range(SIM_NUM):
        i = i0
        while i != n:  # until terminal state is reached
            u = policy[i] - 1  # convert 1-based policy to 0-based index
            sim_costs[s] += G[i, u]

            # Sample next state from cumulative distribution
            r = np.random.rand()
            i = np.searchsorted(PP[i, :, u], r)

    return sim_costs

