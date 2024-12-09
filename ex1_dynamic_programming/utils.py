import numpy as np
import casadi as ca
import scipy.linalg
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Callable, Union, Optional, Tuple, Any
from functools import wraps
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

# Add a tutorial to show how to install acados and set interface to python
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel




class Env:
    def __init__(
            self, 
            case: int, 
            init_state: np.ndarray, 
            target_state: np.ndarray,
            symbolic_h: Callable[[ca.SX, int], ca.SX], 
            symbolic_theta: Callable[[ca.Function], ca.Function],
            state_lbs: np.ndarray = None, 
            state_ubs: np.ndarray = None, 
            input_lbs: float = None, 
            input_ubs: float = None
        ) -> None:

        self.case = case  # 1, 2, 3, 4

        self.initial_position = init_state[0]
        self.initial_velocity = init_state[1]
        self.init_state = np.array([self.initial_position, self.initial_velocity])

        self.target_position = target_state[0]
        self.target_velocity = target_state[1]
        self.target_state = np.array([self.target_position, self.target_velocity])

        self.state_lbs = state_lbs
        self.state_ubs = state_ubs
        self.input_lbs = input_lbs
        self.input_ubs = input_ubs

        # Define argument p as CasADi symbolic parameters
        p = ca.SX.sym("p") # p: horizontal displacement

        # Set functions h(p) and theta(p) based on symbolic parameters
        self.h = ca.Function("h", [p], [symbolic_h(p, case)]) # function of height h w.r.t. p
        self.theta = symbolic_theta(self.h) # function of inclination angle theta w.r.t. p
        
        # Generate grid mesh along p to visualize the curve of slope and inclination angle
        self.p_vals_disp = np.linspace(self.initial_position-0.2, self.target_position+0.2, 50)

    def test_env(self) -> None:
        
        # Calculate values of h and theta on grid mesh
        h_vals = [float(self.h(p)) for p in self.p_vals_disp]
        theta_vals = [float(self.theta(p)) for p in self.p_vals_disp]
        
        # Calculate teh value of h for initial state and terminal state
        initial_h = float(self.h(self.initial_position).full().flatten()[0])
        target_h = float(self.h(self.target_position).full().flatten()[0])
        
        # Display curve theta(p) (left), h(p) (right)
        _, ax = plt.subplots(1, 2, figsize=(12, 3))

        # h(p)
        ax[0].plot(self.p_vals_disp, h_vals, label="h(p)", color='green')
        ax[0].scatter([self.initial_position], [initial_h], color="blue", label="Initial position")
        ax[0].scatter([self.target_position], [target_h], color="orange", label="Target position")
        ax[0].set_xlabel("p")
        ax[0].set_ylabel("h")
        ax[0].set_title("h(p)")
        ax[0].legend()

        # theta(p)
        ax[1].plot(self.p_vals_disp, theta_vals, label=r"$\theta(p)$", color='blue')
        ax[1].set_xlabel("p")
        ax[1].set_ylabel(r"$\theta$")
        ax[1].set_ylim(-1.5, 1.5)  
        ax[1].set_title(r"$\theta$($p$)")
        ax[1].legend()

        plt.show()




class Dynamics:
    def __init__(
            self, 
            state_names: List[str],
            input_names: List[str],
            dynamics_eq: Callable[[ca.Function], Callable[[ca.SX, ca.SX], ca.SX]],
            env: Env
        ) -> None:

        # Define state and input as CasADi symbolic parameters
        self.states = ca.vertcat(*[ca.SX.sym(name) for name in state_names])
        self.inputs = ca.vertcat(*[ca.SX.sym(name) for name in input_names])

        self.dim_states = self.states.shape[0]
        self.dim_inputs = self.inputs.shape[0]
        
        # Define system dynamics
        self.dynamics_function = dynamics_eq(env.theta)

        # Generate symbolic function of linearized Matrix A_lin and B_lin
        lhs = self.dynamics_function(self.states, self.inputs)
        A_sym = ca.jacobian(lhs, self.states)  # Partial derivatives of dynamics w.r.t. states
        B_sym = ca.jacobian(lhs, self.inputs)  # Partial derivatives of dynamics w.r.t. input
        self.A_func = ca.Function("A_func", [self.states, self.inputs], [A_sym])
        self.B_func = ca.Function("B_func", [self.states, self.inputs], [B_sym])

    def get_linearized_AB_discrete(
            self, 
            current_state: np.ndarray, 
            current_input: np.ndarray, 
            dt: float
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''use current state, and current input to calculate the linearized state transfer matrix A_lin and input matrix B_lin'''
        
        # Evaluate linearized A and B around set point
        A_c = np.array(self.A_func(current_state, current_input))
        B_c = np.array(self.B_func(current_state, current_input))

        '''
        # Method 1: approximation using forward euler method
        A_d = A_c * dt + np.eye(self.dim_states)
        B_d = B_c * dt
        '''

        # Method 2: using matrix exponential
        # Construct augmented matrix for continious dynamics
        aug_matrix = np.zeros((self.dim_states + self.dim_inputs, self.dim_states + self.dim_inputs))
        aug_matrix[:self.dim_states, :self.dim_states] = A_c
        aug_matrix[:self.dim_states, self.dim_states:] = B_c

        # Calculate matrix exponential
        exp_matrix = scipy.linalg.expm(aug_matrix * dt)

        # Extract discrete A and B
        A_d = exp_matrix[:self.dim_states, :self.dim_states]
        B_d = exp_matrix[:self.dim_states, self.dim_states:]
        
        # Check controllability of the discretized system
        controllability_matrix = np.hstack([np.linalg.matrix_power(A_d, i) @ B_d for i in range(self.dim_states)])
        rank = np.linalg.matrix_rank(controllability_matrix)
        if rank < self.dim_states:
            raise ValueError(f"System (A, B) is not controllable，rank of controllability matrix is {rank}, while the dimension of state space is {self.dim_states}.")
        
        return A_d, B_d
    
    def compute_equilibrium_input(self, state: np.ndarray) -> float:

        p, v = state

        a = ca.SX.sym("a")
        state_sym = ca.vertcat(ca.SX.sym("p"), ca.SX.sym("v"))
        
        # Define equilibrium_condition: dv/dt = 0
        dynamics_output = self.dynamics_function(state_sym, ca.vertcat(a))
        dvdt = dynamics_output[1] # extract v_dot

        # Substitue the value into symbolic variable to get equilibrium equation
        equilibrium_condition = ca.substitute(dvdt, state_sym, ca.vertcat(p, v))

        # Solve equilibrium equation to get a_eq
        a_eq = ca.solve(equilibrium_condition, a)

        return float(a_eq)
    
    def one_step_forward(self, current_state: np.ndarray, current_input: np.ndarray, dt: float) -> np.ndarray:

        '''use current state, current input, and time difference to calculate next state'''
        
        t_span = [0, dt]
        sim_dynamics = lambda t, state: self.dynamics_function(state, [current_input]).full().flatten()

        solution = solve_ivp(
            sim_dynamics,
            t_span,
            current_state,
            method='RK45',
            t_eval=[dt]
        )
        
        next_state = np.array(solution.y)[:, -1]
        #print(next_state)

        return next_state




def is_square(matrix: np.ndarray) -> bool:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Warning: must be a square matrix!")
    return True

def is_symmetric(matrix: np.ndarray) -> bool:
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Warning: must be a symmetric matrix!")
    return True

def is_positive_semi_definite(matrix: np.ndarray) -> bool:
    eigvals = np.linalg.eigvals(matrix)
    if not np.all(np.greater_equal(eigvals, 0)):
        raise ValueError("Warning: must be a positive semi-definite matrix!")
    return True

def is_positive_definite(matrix: np.ndarray) -> bool:
    eigvals = np.linalg.eigvals(matrix)
    if not np.all(np.greater(eigvals, 0)):
        raise ValueError("Warning: must be a positive definite matrix!")
    return True

def check_input_constraints(compute_action):
    """
    Decorator, for checking whether input is compatible with given constraints
    input constraint get from self.env
    """
    @wraps(compute_action)
    def wrapper(self, current_state, current_time):

        # Get upper and lower bound of input from env
        input_lbs = self.env.input_lbs
        input_ubs = self.env.input_ubs

        # Call original method 'compute_action'
        u = compute_action(self, current_state, current_time)

        # Check whether input is compatible with given constraints
        if input_lbs is not None:
            if not np.all(input_lbs<= u):
                raise ValueError(f"Warning: control input {u} is beyond the lower limit {input_lbs}!")
        if input_ubs is not None:
            if not np.all(u <= input_ubs):
                raise ValueError(f"Warning: control input {u} is beyond the upper limit {input_ubs}")

        return u
    return wrapper


class BaseController(ABC):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            freq: float, 
            name: str, 
            verbose: bool = False
        ) -> None:

        self.name = name

        self.env = env
        self.dynamics = dynamics

        self.freq = freq
        self.dt = 1 / self.freq

        self.verbose = verbose

        self.dim_states = self.dynamics.dim_states
        self.dim_inputs = self.dynamics.dim_inputs

        self.init_state = self.env.init_state
        self.target_state = self.env.target_state

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize necessary matrices or parameters for the controller.
        """
        pass

    @abstractmethod
    def compute_action(self, current_state: np.ndarray, current_time) -> np.ndarray:
        """
        Compute control action based on the current state and time.
        """
        pass


# Derived class for DP Controller
class DPController(BaseController):
    def __init__(
            self,
            env: Env,
            dynamics: Dynamics,
            Q: np.ndarray,
            R: np.ndarray,
            Qf: np.ndarray, 
            freq: float,
            name: str = 'DP',
            Horizon: int = 500,
            verbose: bool = False
        ) -> None:

        super().__init__(env, dynamics, freq, name, verbose)

        self.Q = Q
        self.R = R
        self.Qf = Qf

        self.N = Horizon
        
        # Different from LQR or iLQR, there we directly use self.policy to store inputs
        self.policy = [None] * self.N

        self.setup()

    def setup(self) -> None:

        # Initialize Acados model
        model = AcadosModel()
        model.name = "dp_controller"
        model.x = self.dynamics.states
        model.u = self.dynamics.inputs
        model.f_expl_expr = ca.vertcat(self.dynamics.dynamics_function(self.dynamics.states, self.dynamics.inputs))
        model.f_impl_expr = None

        # Set up Acados OCP
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N  # Prediction horizon
        ocp.solver_options.tf = self.N * self.dt  # Total prediction time
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.globalization = 'FUNNEL_L1PEN_LINESEARCH'
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.print_level = 1
        ocp.solver_options.nlp_solver_tol_stat = 1e-5

        # Set up cost function
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = np.block([
            [self.Q, np.zeros((self.dim_states, self.dim_inputs))],
            [np.zeros((self.dim_inputs, self.dim_states)), self.R],
        ])
        ocp.cost.W_e = self.Qf # will be updated in each step

        # Set up mapping from QP to OCP
        # Define output matrix for non-terminal state
        ocp.cost.Vx = np.block([
            [np.eye(self.dim_states)],
            [np.zeros((self.dim_inputs, self.dim_states))]
        ])
        # Define breakthrough matrix for non-terminal state
        ocp.cost.Vu = np.block([
            [np.zeros((self.dim_states, self.dim_inputs))],
            [np.eye(self.dim_inputs)]
        ])
        # Define output matrix for terminal state
        ocp.cost.Vx_e = np.eye(self.dim_states)

        # Initialize reference of task (stabilization)
        ocp.cost.yref = np.hstack((self.target_state, np.zeros(self.dim_inputs)))
        ocp.cost.yref_e = np.array(self.target_state)

        # Define constraints
        ocp.constraints.x0 = self.init_state  # Initial state
        
        ocp.constraints.idxbx = np.arange(self.dim_states)
        ocp.constraints.idxbx_e = np.arange(self.dim_states)
        if self.env.state_lbs is None and self.env.state_ubs is None:
            ocp.constraints.lbx = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx_e = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_e = np.full(self.dim_states, 1e6)
        elif self.env.state_lbs is not None and self.env.state_ubs is None:
            ocp.constraints.lbx = np.array(self.env.state_lbs)
            ocp.constraints.ubx = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx_e = np.array(self.env.state_lbs)
            ocp.constraints.ubx_e = np.full(self.dim_states, 1e6)
        elif self.env.state_lbs is None and self.env.state_ubs is not None:
            ocp.constraints.lbx = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx = np.array(self.env.state_ubs)
            ocp.constraints.lbx_e = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        else:
            ocp.constraints.lbx = np.array(self.env.state_lbs)
            ocp.constraints.ubx = np.array(self.env.state_ubs)
            ocp.constraints.lbx_e = np.array(self.env.state_lbs)
            ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        
        ocp.constraints.idxbu = np.arange(self.dim_inputs)
        if self.env.input_lbs is None and self.env.input_ubs is None:
            ocp.constraints.lbu = np.full(self.dim_inputs, -1e6)
            ocp.constraints.ubu = np.full(self.dim_inputs, 1e6)
        elif self.env.input_lbs is not None and self.env.input_ubs is None:
            ocp.constraints.lbu = np.array(self.env.input_lbs)
            ocp.constraints.ubu = np.full(self.dim_inputs, 1e6)
        elif self.env.input_lbs is None and self.env.input_ubs is not None:
            ocp.constraints.lbu = np.full(self.dim_inputs, -1e6)
            ocp.constraints.ubu = np.array(self.env.input_ubs)
        else:
            ocp.constraints.lbu = np.array(self.env.input_lbs)
            ocp.constraints.ubu = np.array(self.env.input_ubs)

        # Set up Acados solver
        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file=f"{model.name}.json")

        if self.verbose:
            print("DP setup with Acados completed.")

        # Update initial state in the solver
        self.solver.set(0, "lbx", self.init_state)
        self.solver.set(0, "ubx", self.init_state)

        # Solve the MPC problem
        status = self.solver.solve()
        if status != 0:
            raise ValueError(f"Acados solver failed with status {status}")

        # Extract each input 
        for k in range(self.N):

            self.policy[k] = self.solver.get(k, "u")

            print(f"Current input: {self.policy[k]}")

        if self.verbose:
            print(f"Dynamic Programming policy with input constraints computed.")

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_time: int) -> np.ndarray:

        if self.policy is None:
            raise ValueError("DP Policy is not computed. Call setup() first.")

        # Discretize the current time to find the policy index
        time_index = int(np.floor(current_time * self.freq))
        if time_index >= self.N:
            time_index = self.N - 1  # Use the last policy (zero policy) for any time after the horizon

        return self.policy[current_time]


# Derived class for LQR Controller
class LQRController(BaseController):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            freq: float, 
            name: str = 'LQR', 
            verbose: bool = False
        ) -> None:

        super().__init__(env, dynamics, freq, name, verbose)
        
        # Initialize as private property
        self._Q = None
        self._R = None
        self._K = None  # LQR gain matrix
        
        # Call setter for the check and update the value of private property
        self.Q = Q
        self.R = R

        self.A = None  # State transfer matrix
        self.B = None  # Input matrix
        
        # need no external objects for setup, can directly call the setup function here
        if name in ['LQR', 'MPC']:
            self.setup()

    @property
    def Q(self) -> np.ndarray:
        return self._Q

    @Q.setter
    def Q(self, value: np.ndarray) -> None:

        is_square(value)
        is_symmetric(value)
        is_positive_semi_definite(value)

        if self.verbose:
            print("Check passed, Q is a symmetric, positive semi-definite matrix.")

        self._Q = value

    @property
    def R(self) -> np.ndarray:
        return self._R

    @R.setter
    def R(self, value: np.ndarray) -> None:

        is_square(value)
        is_symmetric(value)
        is_positive_definite(value)

        if self.verbose:
            print("Check passed, R is a symmetric, positive definite matrix.")
        
        self._R = value
    
    @property
    def K(self) -> np.ndarray:
        return self._K

    @K.setter
    def K(self, value: np.ndarray) -> None:
        eigvals = np.linalg.eigvals(self.A - self.B @ value)
        if np.any(np.abs(eigvals) > 1):
            raise ValueError("Warning: not all eigenvalue of A_cl inside unit circle, close-loop system is unstable!")
        elif self.verbose:
            print(f"Check passed, current gain K={value}, close-loop system is stable.")
        self._K = value

    def setup(self) -> None:
        # Linearize dynamics at equilibrium
        state_eq = np.zeros(self.dim_states)
        input_eq = np.zeros((1,))
        self.A, self.B = self.dynamics.get_linearized_AB_discrete(
            current_state=state_eq, current_input=input_eq, dt=self.dt
        )

        # Solve DARE to compute gain matrix
        P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

        if self.verbose:
            print(f"LQR Gain Matrix K: {self.K}")

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_time) -> np.ndarray:
        if self.K is None:
            raise ValueError("LQR gain matrix K is not computed. Call setup() first.")

        # Compute state error
        det_x = current_state - self.target_state

        # Apply control law
        u = -self.K @ det_x
        return u


# Derived class for iLQR Controller
class iLQRController(LQRController):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Qf: np.ndarray, 
            freq: float, 
            name: str = 'iLQR', 
            max_iter: int = 30, 
            tol: float = 2e-2, 
            verbose: bool = False
        ) -> None:

        self.Qf = Qf  # Terminal cost matrix

        self.max_iter = max_iter
        self.tol = tol

        self.K_k_arr = None
        self.u_kff_arr = None

        super().__init__(env, dynamics, Q, R, freq, name, verbose)

    def setup(self, input_traj: np.ndarray) -> None:
        """
        Perform iLQR to compute the optimal control sequence.
        """
        
        N = len(input_traj)

        # Initialize state and control trajectories
        x_traj = np.zeros((self.dim_states, N+1))  # State trajector
        u_traj = np.copy(input_traj)  # Control trajectory
        x_traj[:, 0] = self.init_state  # Initial state
        
        # Initialize fb and ff gain
        self.K_k_arr = np.zeros((self.dim_states, N))
        self.u_kff_arr = np.zeros((N))

        for n in range(self.max_iter):

            # Forward pass: Simulate system using current control sequence
            for k in range(N):
                next_state = self.dynamics.one_step_forward(current_state=x_traj[:, k], current_input=u_traj[k], dt=self.dt)
                x_traj[:, k + 1] = next_state

            # Backward pass: Compute cost-to-go and update control
            x_N_det = x_traj[:, -1] - self.target_state
            x_N_det = x_N_det.reshape(-1, 1) # reshape into column vector
            #print(f"x_N_det: {x_N_det}")

            s_k_bar = (x_N_det.T @ self.Qf @ x_N_det) / 2 # Terminal cost
            s_k = self.Qf @ x_N_det # Terminal cost gradient
            S_k = self.Qf # Terminal cost Hessian

            for k in range(N - 1, -1, -1):

                # Linearize dynamics: f(x, u) ≈ A*x + B*u
                A_lin, B_lin = self.dynamics.get_linearized_AB_discrete(current_state=x_traj[:, k], current_input=u_traj[k], dt=self.dt)

                # Compute Q matrices
                x_k_det = x_traj[:, k] - self.target_state
                x_k_det = x_k_det.reshape(-1, 1) # reshape into column vector
                
                g_k_bar = (x_k_det.T @ self.Q @ x_k_det + self.R * u_traj[k] ** 2) * self.dt / 2
                q_k = (self.Q @ x_k_det) * self.dt
                Q_k = (self.Q) * self.dt
                r_k = (self.R * u_traj[k]) * self.dt
                R_k = (self.R) * self.dt
                P_k = np.zeros((2,)) * self.dt # should be row vector

                l_k = (r_k + B_lin.T @ s_k)
                G_k = (P_k + B_lin.T @ S_k @ A_lin) # should be row vector
                H_k = (R_k + B_lin.T @ S_k @ B_lin)

                det_u_kff = - np.linalg.inv(H_k) @ l_k
                K_k = - np.linalg.inv(H_k) @ G_k  # should be row vector
                u_kff = u_traj[k] + det_u_kff - (K_k @ x_traj[:, k])

                self.K_k_arr[:, k] = (K_k.T).flatten()
                self.u_kff_arr[k] = u_kff.item()

                s_k_bar = g_k_bar + s_k_bar + (det_u_kff.T @ H_k @ det_u_kff) / 2 + det_u_kff.T @ l_k
                s_k = q_k + A_lin.T @ s_k + K_k.T @ H_k @ det_u_kff + K_k.T @ l_k + G_k.T @ det_u_kff
                S_k = Q_k + A_lin.T @ S_k @ A_lin + K_k.T @ H_k @ K_k + K_k.T @ G_k + G_k.T @ K_k

                if self.verbose:
                    print(f"s_k_bar: {s_k_bar}")
                    print(f"s_k: {s_k}")
                    print(f"S_k: {S_k}")

                    print(f"A_lin: {A_lin}")
                    print(f"B_lin: {B_lin}")

                    print(f"x_k_det: {x_k_det}")

                    print(f"g_k_bar: {g_k_bar}")
                    print(f"q_k: {q_k}")
                    print(f"Q_k: {Q_k}")
                    print(f"r_k: {r_k}")
                    print(f"R_k: {R_k}")
                    print(f"P_k: {P_k}")

                    print(f"l_k: {l_k}")
                    print(f"G_k: {G_k}")
                    print(f"H_k: {H_k}")

                    print(f"det_u_kff: {det_u_kff}")
                    print(f"K_k: {K_k}")

                    print(f"A_lin.T @ S_k @ A_lin: {A_lin.T @ S_k @ A_lin}")
                    print(f"A_lin.T @ s_k: {A_lin.T @ s_k}")
                    print(f"K_k.T @ H_k @ K_k: {K_k.T @ H_k @ K_k}")
                    print(f"G_k.T @ K_k: {G_k.T @ K_k}")
                
            # Update control sequence
            new_u_traj = np.zeros_like(u_traj)
            new_x_traj = np.zeros_like(x_traj)
            new_x_traj[:, 0] = self.init_state
            
            # Simulation forward to get input sequence
            for k in range(N):
                new_u_traj[k] = self.u_kff_arr[k] + self.K_k_arr[:, k].T @ new_x_traj[:, k]
                next_state = self.dynamics.one_step_forward(current_state=new_x_traj[:, k], current_input=new_u_traj[k], dt=self.dt)
                new_x_traj[:, k + 1] = next_state

            # Check for convergence
            if np.max(np.abs(new_u_traj - u_traj)) < self.tol:
                print(f"Use {n} iteration until converge.")
                break
            else:
                print(f"Iteration {n}: residual error is {np.max(np.abs(new_u_traj - u_traj))}")
                #print(f"Old input trajectory: {u_traj.flatten()}")
                #print(f"New input trajectory: {new_u_traj.flatten()}")

            u_traj = new_u_traj
            x_traj = new_x_traj

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_time: int) -> np.ndarray:
        if self.K_k_arr is None or self.u_kff_arr is None:
            raise ValueError("iLQR parameters are not computed. Call setup() first.")

        u = self.u_kff_arr[current_time] + self.K_k_arr[:, current_time].T @ current_state
        return u


# Derived class for MPC Controller
class MPCController(LQRController):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Qf: np.ndarray, 
            freq: float, 
            N: int, 
            name: str = 'MPC', 
            verbose: bool = True
        ) -> None:

        """
        Initialize the MPC Controller with Acados.

        Args:
        - env: The environment providing initial and target states.
        - dynamics: The system dynamics.
        - Q: State cost matrix.
        - R: Control cost matrix.
        - Qf: Terminal state cost matrix.
        - freq: Control frequency.
        - N: Prediction horizon.
        - verbose: Print debug information if True.
        """

        self.Qf = Qf

        self.N = N  # Prediction horizon

        self.ocp = None
        self.solver = None

        super().__init__(env, dynamics, Q, R, freq, name, verbose)

    def setup(self) -> None:
        """
        Define the MPC optimization problem using Acados.
        """
        # Initialize Acados model
        model = AcadosModel()
        model.name = "mpc_controller"

        # Define model: dx/dt = f(x, u)
        model.x = self.dynamics.states
        model.u = self.dynamics.inputs
        model.f_expl_expr = ca.vertcat(self.dynamics.dynamics_function(self.dynamics.states, self.dynamics.inputs))
        model.f_impl_expr = None

        # Set up Acados OCP
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N  # Prediction horizon
        ocp.solver_options.tf = self.N * self.dt  # Total prediction time
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP"

        # Set up cost function
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.W = np.block([
            [self.Q, np.zeros((self.dim_states, self.dim_inputs))],
            [np.zeros((self.dim_inputs, self.dim_states)), self.R],
        ])
        ocp.cost.W_e = self.Qf

        # Set up mapping from QP to OCP
        # Define output matrix for non-terminal state
        ocp.cost.Vx = np.block([
            [np.eye(self.dim_states)],
            [np.zeros((self.dim_inputs, self.dim_states))]
        ])
        # Define breakthrough matrix for non-terminal state
        ocp.cost.Vu = np.block([
            [np.zeros((self.dim_states, self.dim_inputs))],
            [np.eye(self.dim_inputs)]
        ])
        # Define output matrix for terminal state
        ocp.cost.Vx_e = np.eye(self.dim_states)

        # Initialize reference of task (stabilization)
        ocp.cost.yref = np.zeros(self.dim_states + self.dim_inputs) 
        ocp.cost.yref_e = np.zeros(self.dim_states) 

        # Define constraints
        ocp.constraints.x0 = self.init_state  # Initial state
        
        ocp.constraints.idxbx = np.arange(self.dim_states)
        ocp.constraints.idxbx_e = np.arange(self.dim_states)
        if self.env.state_lbs is None and self.env.state_ubs is None:
            ocp.constraints.lbx = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx_e = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_e = np.full(self.dim_states, 1e6)
        elif self.env.state_lbs is not None and self.env.state_ubs is None:
            ocp.constraints.lbx = np.array(self.env.state_lbs)
            ocp.constraints.ubx = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx_e = np.array(self.env.state_lbs)
            ocp.constraints.ubx_e = np.full(self.dim_states, 1e6)
        elif self.env.state_lbs is None and self.env.state_ubs is not None:
            ocp.constraints.lbx = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx = np.array(self.env.state_ubs)
            ocp.constraints.lbx_e = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        else:
            ocp.constraints.lbx = np.array(self.env.state_lbs)
            ocp.constraints.ubx = np.array(self.env.state_ubs)
            ocp.constraints.lbx_e = np.array(self.env.state_lbs)
            ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        
        ocp.constraints.idxbu = np.arange(self.dim_inputs)
        if self.env.input_lbs is None and self.env.input_ubs is None:
            ocp.constraints.lbu = np.full(self.dim_inputs, -1e6)
            ocp.constraints.ubu = np.full(self.dim_inputs, 1e6)
        elif self.env.input_lbs is not None and self.env.input_ubs is None:
            ocp.constraints.lbu = np.array(self.env.input_lbs)
            ocp.constraints.ubu = np.full(self.dim_inputs, 1e6)
        elif self.env.input_lbs is None and self.env.input_ubs is not None:
            ocp.constraints.lbu = np.full(self.dim_inputs, -1e6)
            ocp.constraints.ubu = np.array(self.env.input_ubs)
        else:
            ocp.constraints.lbu = np.array(self.env.input_lbs)
            ocp.constraints.ubu = np.array(self.env.input_ubs)

        # Set up Acados solver
        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file=f"{model.name}.json")

        if self.verbose:
            print("MPC setup with Acados completed.")

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_time) -> np.ndarray:
        """
        Solve the MPC problem and compute the optimal control action.

        Args:
        - current_state: The current state of the system.
        - current_time: The current time (not directly used).

        Returns:
        - Optimal control action.
        """
        # Update initial state in the solver
        self.solver.set(0, "lbx", current_state)
        self.solver.set(0, "ubx", current_state)

        # Update reference trajectory for all prediction steps
        state_ref = self.target_state
        input_ref = np.zeros(self.dim_inputs)
        for i in range(self.N):
            self.solver.set(i, "yref", np.concatenate((state_ref, input_ref)))
        self.solver.set(self.N, "yref", state_ref) # set reference valur for y_N seperately (different shape)

        # Solve the MPC problem
        status = self.solver.solve()
        if status != 0:
            raise ValueError(f"Acados solver failed with status {status}")

        # Extract the first control action
        u_optimal = self.solver.get(0, "u")

        if self.verbose:
            print(f"Optimal control action: {u_optimal}")

        return u_optimal




class Simulator:
    def __init__(
            self, 
            dynamics: Dynamics, 
            controller: BaseController, 
            env: Env, 
            dt: float, 
            t_terminal: float, 
            verbose: bool = False
        ) -> None:

        self.dynamics = dynamics
        self.controller = controller
        self.env = env

        self.init_state = self.env.init_state
        
        # Define timeline
        self.t_0 = 0
        self.t_terminal = t_terminal
        self.dt = dt
        self.t_eval = np.linspace(self.t_0, self.t_terminal, int((self.t_terminal - self.t_0) / self.dt))
        
        # Initialize recording list for state and input sequence
        self.state_traj = [] # list -> ndarray(2,)
        self.input_traj = [] # list -> scalar

        self.positions = []
        self.velocities = []
        self.accelerations = []
        
        # Initialize counter to present current time
        self.counter = 0

        self.verbose = verbose

    def reset_counter(self) -> None:
        self.counter = 0
    
    def run_simulation(self) -> None:
        
        # Initialize state vector
        current_state = self.init_state

        for current_time in self.t_eval:

            # Get current state, and call controller to calculate input
            current_input = self.controller.compute_action(current_state, self.counter)

            # Do one-step simulation
            current_state = self.dynamics.one_step_forward(current_state, current_input, self.dt)
            #print(f"sim_state:{current_state}")

            # Log the results
            self.state_traj.append(current_state)
            self.input_traj.append(current_input)

            # Update timer
            self.counter += 1
        
        print("Simulation finished, will start plotting")
        self.reset_counter()

    def get_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Get state and input trajectories, return in ndarray form '''

        # Transform state and input traj from list to ndarray
        state_traj = np.array(self.state_traj)
        input_traj = np.array(self.input_traj)

        return state_traj, input_traj




class Visualizer:

    def __init__(self, simulator: Simulator) -> None:

        self.simulator = simulator
        self.dynamics = simulator.dynamics
        self.controller = simulator.controller
        self.env = simulator.env

        self.state_traj, self.input_traj = self.simulator.get_trajectories()

        self.position = self.state_traj[:, 0]
        self.velocity = self.state_traj[:, 1]
        self.acceleration = self.input_traj
        
        self.dt = simulator.dt
        self.t_eval = simulator.t_eval

        # Define setting of animation
        self.car_length= 0.2
        self.figsize = (8, 4)
        self.refresh_rate = 30

    def display_final_results(self) -> None:

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))

        # Plot p / v / a over time t
        ax1.plot(self.t_eval, self.position, label="Position p(t)", color="blue")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.legend()

        # Plot p / v / a over time t
        ax2.plot(self.t_eval, self.velocity, label="Velocity v(t)", color="green")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.legend()

        # Plot p / v / a over time t
        ax3.plot(self.t_eval, self.acceleration, label="Acceleration a(t)", color="red")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Acceleration (m/s^2)")
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def display_contrast(self, simulator_ref: Simulator) -> None:
        
        # Get reference trajectories from simulator_ref
        state_traj_ref, input_traj_ref = simulator_ref.get_trajectories()
        
        # Extract reference position velocity and acceleration
        position_ref = state_traj_ref[:, 0]
        velocity_ref = state_traj_ref[:, 1]
        acceleration_ref = input_traj_ref

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))

        # Plot p / v / a over time t
        ax1.plot(self.t_eval, self.position, label=self.controller.name, color="blue")
        ax1.plot(self.t_eval, position_ref, linestyle="--", label=simulator_ref.controller.name, color="blue")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.legend()

        # Plot p / v / a over time t
        ax2.plot(self.t_eval, self.velocity, label=self.controller.name, color="green")
        ax2.plot(self.t_eval, velocity_ref, linestyle="--", label=simulator_ref.controller.name, color="green")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.legend()

        # Plot p / v / a over time t
        ax3.plot(self.t_eval, self.acceleration, label=self.controller.name, color="red")
        ax3.plot(self.t_eval, acceleration_ref, linestyle="--", label=simulator_ref.controller.name, color="red")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Acceleration (m/s^2)")
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def display_animation(self) -> HTML:
        
        # Instantiate the plotting
        fig, ax1 = plt.subplots(1, 1, figsize=self.figsize)
        
        # Define size of plotting
        p_max = max(self.position)
        p_min = min(self.position)
        start_extension = p_min - 0.3
        end_extension = p_max + 0.3

        p_disp_vals = np.linspace(start_extension, end_extension, 50) # generate grid mesh on p

        h_disp_vals = [float(self.env.h(p).full().flatten()[0]) for p in p_disp_vals]
        h_max = max(h_disp_vals)
        h_min = min(h_disp_vals)

        ax1.set_xlim(start_extension-0.5, end_extension+0.5)
        ax1.set_ylim(h_min-0.5, h_max+0.5)

        # Draw mountain profile curve h(p)
        ax1.plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="green")
        ax1.set_xlabel("Position p")
        ax1.set_ylabel("Height h")
        ax1.legend()

        # Mark the intial state and the target state in the plotting
        initial_h = float(self.env.h(self.env.initial_position).full().flatten()[0])
        target_h = float(self.env.h(self.env.target_position).full().flatten()[0])

        ax1.scatter([self.env.initial_position], [initial_h], color="blue", label="Initial position")
        ax1.scatter([self.env.target_position], [target_h], color="orange", label="Target position")
        ax1.legend()


        # Setting simplyfied car model as rectangle, and update the plotting to display the animation
        car_height = self.car_length / 2
        car = Rectangle((0, 0), self.car_length, car_height, color="red")
        ax1.add_patch(car)

        def update(frame):
            # Get current position and attitude of car
            current_position = self.position[frame]
            current_theta = float(self.env.theta(current_position).full().flatten()[0])

            # Update position and attitude of car
            car.set_xy((current_position, float(self.env.h(current_position).full().flatten()[0])))
            car.angle = np.degrees(current_theta)  # rad to deg

        # Instantiate animation
        anim = FuncAnimation(fig, update, frames=len(self.t_eval), interval=1000 / self.refresh_rate, repeat=False)

        return HTML(anim.to_jshtml())



# Actions:
# split up the controller and simulator
# add property decorator to limit the charactor of arguments (https://stackoverflow.com/questions/17330160/how-does-the-property-decorator-work-in-python)
# - conrtollability for A_d and B_d
# - symmetric for Q and R, psdf for Q, pdf for R
# - stability for close-loop gain
# - whether compatible with constraint for state / input
# add a dataclass to set up all internal & public variable in class (https://docs.python.org/3/library/dataclasses.html)


# Next steps:
# finish PID
# implement ADP (without constraint)
