import sympy as sp
import pickle
import time

class SecondOrderDPSolver:
    def __init__(self, N, Q, R, Qf, x_ref):
        """
        DP solver for linear system:
            x_{k+1} = A x_k + B u_k
            cost = sum u_k^T R u_k + terminal cost
        """
        self.N = N
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.x_ref = sp.Matrix(x_ref)

        # A, B matrices for 2nd order system
        # siqi's case:
        #self.A = sp.Matrix([[1, 0], [0, 0]])
        #self.B = sp.Matrix([[1], [0]])
        # mountain car case:
        self.A = sp.Matrix([[1, 1], [0, 1]])
        self.B = sp.Matrix([[0.5], [1]])

        # Create symbolic variables
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

        # Terminal cost: J_N = (x_N - x_ref)^T Qf (x_N - x_ref)
        err_N = x[N] - x_ref
        J[N] = (err_N.T * Qf * err_N)[0, 0]

        for k in reversed(range(N)):
            # x_{k+1} = A x_k + B u_k
            x_next = A * x[k] + B * u[k]

            # Cost at step k: u_k^T R u_k + J_{k+1}(x_{k+1})
            stage_cost = R * u[k]**2
            J_kplus1_sub = J[k + 1].subs({x[k + 1][i]: x_next[i] for i in range(2)})

            total_cost = stage_cost + J_kplus1_sub

            # Derivative w.r.t u_k
            dJ_du = sp.diff(total_cost, u[k])
            u_star = sp.solve(dJ_du, u[k])[0]
            mu[k] = sp.simplify(u_star)

            # Plug u_k* back into cost to get J_k
            cost_k_opt = total_cost.subs(u[k], u_star)
            J[k] = sp.simplify(cost_k_opt)

            print(f"i = {N-k}")
            print(f"Pass time = {time.time() - self.start_time:.2f} seconds")
            print(f"u_{k}*(x_{k}) =", mu[k])

        self.J = J
        self.mu = mu

    def print_solution(self):
        for k, uk in enumerate(self.mu):
            print(f"u_{k}*(x_{k}) =", uk)
        #print("\nJ_0(x_0) =")
        #sp.pprint(self.J[0])

    def save_policy(self, filename='dp_policy.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({
                'mu': self.mu,
                'J': self.J,
            }, f)

    def load_policy(self, filename='dp_policy.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.mu = data['mu']
            self.J = data['J']



if __name__ == "__main__":

    N = 10  # Horizon

    # Define symbols
    q1, q2 = sp.symbols('q_p q_v')
    Q = sp.diag(q1, q2)
    R = sp.Symbol('r')
    Qf = sp.diag(q1, q2)  # same terminal cost
    p_ref, v_ref = sp.symbols('p_ref v_ref')
    x_ref = [p_ref, v_ref]

    solver = SecondOrderDPSolver(N, Q, R, Qf, x_ref)
    solver.solve()
    solver.print_solution()

    solver.save_policy('dp_policy_test.pkl')

    solver_new = SecondOrderDPSolver(N, Q, R, Qf, x_ref)
    solver_new.load_policy('dp_policy_test.pkl')
    solver_new.print_solution()