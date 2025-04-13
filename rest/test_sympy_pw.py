import sympy as sp
import numpy as np
import time

class SecondOrderDPSolver:
    def __init__(self, N, Q, R, Qf, x_ref):
        """
        DP solver for linear system with discrete input: u ∈ {-1, 1}
        Dynamics: x_{k+1} = A x_k + B u_k
        Cost: sum u_k^T R u_k + terminal cost
        """
        self.N = N
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.x_ref = sp.Matrix(x_ref)

        # A, B matrices for 2nd order system (e.g. mountain car)
        # siqi's case:
        #self.A = sp.Matrix([[1, 0], [0, 0]])
        #self.B = sp.Matrix([[1], [0]])
        # mountain car case:
        self.A = sp.Matrix([[1, 1], [0, 1]])
        self.B = sp.Matrix([[0.5], [1]])

        # Create symbolic variablesS
        self.x = [sp.Matrix(sp.symbols(f'p_{k} v_{k}')) for k in range(N + 1)]
        self.u = [sp.Symbol(f'u_{k}') for k in range(N)]

        # Store results
        self.J = [None] * (N + 1)
        self.mu = [None] * N

        # start timing
        self.start_time = time.time()

    def solve(self):
        N, Q, R, Qf, x_ref = self.N, self.Q, self.R, self.Qf, self.x_ref
        A, B, x, u = self.A, self.B, self.x, self.u
        J, mu = self.J, self.mu

        # Terminal cost
        err_N = x[N] - x_ref
        J[N] = (err_N.T * Qf * err_N)[0, 0]

        for k in reversed(range(N)):

            J_next = J[k + 1]
            if isinstance(J_next, sp.Piecewise):
                branches = J_next.args
            else:
                branches = [(J_next, True)]

            mu_branches = []
            J_branches = []

            for expr_next, cond in branches:
                # Evaluate only at u = -1 and u = +1
                x_next_m1 = A * x[k] + B * (-1)
                x_next_p1 = A * x[k] + B * (1)

                Jp1_m1 = expr_next.subs({x[k+1][i]: x_next_m1[i] for i in range(2)})
                Jp1_p1 = expr_next.subs({x[k+1][i]: x_next_p1[i] for i in range(2)})

                cost_m1 = R * 1 + Jp1_m1
                cost_p1 = R * 1 + Jp1_p1

                delta = cost_p1 - cost_m1

                mu_branch = sp.Piecewise(
                    (-1, delta > 0),
                    (1, True)
                )
                J_branch = sp.Piecewise(
                    (cost_m1, delta > 0),
                    (cost_p1, True)
                )

                mu_branches.append((mu_branch, cond))
                J_branches.append((J_branch, cond))

            mu[k] = sp.Piecewise(*mu_branches)
            J[k] = sp.Piecewise(*J_branches)

            print(f"i = {N-k}")
            print(f"Pass time = {time.time() - self.start_time:.2f} seconds")

        self.mu = mu
        self.J = J



    def print_solution(self):
        print("Optimal control policy μ_k(x_k):")
        for k, uk in enumerate(self.mu):
            print(f"\nu_{k}*(x_{k}) =")
            sp.pprint(uk)
        #print("\nCost-to-go from initial state:")
        #sp.pprint(self.J[0])


if __name__ == "__main__":
    N = 10  # Horizon length

    # Define symbolic weights and reference
    #q1, q2 = sp.symbols('q_p q_v')
    #Q = sp.diag(q1, q2)
    #R = sp.Symbol('r')
    #Qf = Q  # Terminal weight

    Q = sp.Matrix(np.diag([1, 1]))
    R = sp.Float(np.array([[0.1]]))
    Qf = Q
    
    #p_ref, v_ref = sp.symbols('p_ref v_ref')
    #x_ref = [p_ref, v_ref]
    x_ref = sp.Matrix(np.array([[0.5], [0]]))

    # Instantiate and solve
    solver = SecondOrderDPSolver(N, Q, R, Qf, x_ref)
    solver.solve()
    solver.print_solution()
