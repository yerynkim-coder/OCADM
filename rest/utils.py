import numpy as np
import casadi as ca
import scipy.linalg
import copy
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Callable, Union, Optional, Tuple, Any
from functools import wraps
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
from scipy.interpolate import interp2d

# Add a tutorial to show how to install acados and set interface to python
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel




class Env:
    def __init__(
            self, 
            case: int, 
            init_state: np.ndarray, 
            target_state: np.ndarray,
            symbolic_h: Callable[[ca.SX, int], ca.SX] = None, 
            symbolic_theta: Callable[[ca.Function], ca.Function] = None,
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
        if state_lbs is not None:
            self.pos_lbs = state_lbs[0]
            self.vel_lbs = state_lbs[1]
        self.state_ubs = state_ubs
        if state_ubs is not None:
            self.pos_ubs = state_ubs[0]
            self.vel_ubs = state_ubs[1]
        self.input_lbs = input_lbs
        self.input_ubs = input_ubs

        # Define argument p as CasADi symbolic parameters
        p = ca.SX.sym("p") # p: horizontal displacement

        # Initialize symbolic h and theta if not given
        if symbolic_h == None:

            def symbolic_h(p, case):
                if case == 1:  # zero slope
                    h = 0
                elif case == 2: # constant slope
                    h = (ca.pi * p) / 18
                elif case == 3: # varying slope
                    h = ca.cos(3 * p)
                elif case == 4: # varying slope for underactuated case
                    h = ca.sin(3 * p)
                return h
        
        if symbolic_theta == None:

            def symbolic_theta(h_func):
                h = h_func(p) 
                dh_dp = ca.jacobian(h, p)
                theta = ca.atan(dh_dp)
                return ca.Function("theta", [p], [theta])

        # Set functions h(p) and theta(p) based on symbolic parameters
        self.h = ca.Function("h", [p], [symbolic_h(p, case)]) # function of height h w.r.t. p
        self.theta = symbolic_theta(self.h) # function of inclination angle theta w.r.t. p
        
        # Generate grid mesh along p to visualize the curve of slope and inclination angle
        if self.state_lbs is not None:
            lbs_position = self.state_lbs[0]
        else:
            lbs_position = self.initial_position-0.2

        if self.state_ubs is not None:
            ubs_position = self.state_ubs[0]
        else:
            ubs_position = self.target_position+0.2

        self.p_vals_disp = np.linspace(lbs_position, ubs_position, 50)

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
            env: Env,
            state_names: List[str] = None,
            input_names: List[str] = None,
            setup_dynamics: Callable[[ca.Function], Callable[[ca.SX, ca.SX], ca.SX]] = None,
        ) -> None:

        # Initialize system dynmaics if not given
        if state_names is None and input_names is None:
            state_names = ["p", "v"]
            input_names = ["a"]

        # Define state and input as CasADi symbolic parameters
        self.states = ca.vertcat(*[ca.SX.sym(name) for name in state_names])
        self.inputs = ca.vertcat(*[ca.SX.sym(name) for name in input_names])

        self.dim_states = self.states.shape[0]
        self.dim_inputs = self.inputs.shape[0]

        # Initialize system dynmaics if not given
        if setup_dynamics is None:

            def setup_dynamics(theta_function):

                p = ca.SX.sym("p")
                v = ca.SX.sym("v")
                a = ca.SX.sym("a")
                Gravity = 9.81

                theta = theta_function(p)

                # Expression of dynamics
                dpdt = v
                dvdt = a * ca.cos(theta) # - Gravity * ca.sin(theta) * ca.cos(theta)
                
                state = ca.vertcat(p, v)
                input = ca.vertcat(a)
                rhs = ca.vertcat(dpdt, dvdt)

                return ca.Function("dynamics_function", [state, input], [rhs])
        
        # Define system dynamics
        self.dynamics_function = setup_dynamics(env.theta)

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



class Env_rl(Env):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics = None,
            num_states: np.array = np.array([20, 20]),
            num_actions: int = 10,
            dt: float = 0.1
        ) -> None:

        self.env = env
        
        super().__init__(self.env.case, self.env.init_state, self.env.target_state, 
                         state_lbs=self.env.state_lbs, state_ubs=self.env.state_ubs, 
                         input_lbs=self.env.input_lbs, input_ubs=self.env.input_ubs)

        self.num_pos = num_states[0]
        self.num_vel = num_states[1]
        self.num_acc = num_actions

        if self.state_lbs is None or self.state_ubs is None or self.input_lbs is None or self.input_ubs is None:
            raise ValueError("Constraints on states and input must been fully specified!")

        # Partitions over state space and input space
        self.pos_partitions = np.linspace(self.pos_lbs, self.pos_ubs, self.num_pos)
        self.vel_partitions = np.linspace(self.vel_lbs, self.vel_ubs, self.num_vel)
        self.acc_partitions = np.linspace(self.input_lbs, self.input_ubs, self.num_acc)

        # Generate state space (position, velocity combinations)
        POS, VEL = np.meshgrid(self.pos_partitions, self.vel_partitions)
        self.state_space = np.vstack((POS.ravel(), VEL.ravel()))  # Shape: (2, num_states)
        self.num_states = self.state_space.shape[1]

        # Define action space (acceleration)
        self.input_space = self.acc_partitions  # Shape: (1, num_actions)
        self.num_actions = len(self.input_space)
        
        # State propagation
        self.dynamics = dynamics
        self.dt = dt

        # Define transition probability matrix and reward matrix
        self.T = [None] * self.num_actions  # Shape: list (1, num_actions) -> array (num_states, num_states)
        self.R = [None] * self.num_actions  # Shape: list (1, num_actions) -> array (num_states, num_states)
        
        # Build MDP 
        self.build_stochastic_mdp()

    def one_step_forward(self, cur_state, cur_input):
        
        # Check whether current state and input is within the state space and input space
        cur_pos = max(min(cur_state[0], self.pos_ubs), self.pos_lbs)
        cur_vel = max(min(cur_state[1], self.vel_ubs), self.vel_lbs)
        cur_state = np.array([cur_pos, cur_vel])
        cur_input = max(min(cur_input, self.input_ubs), self.input_lbs)
        
        # Propagate the state
        next_state = self.dynamics.one_step_forward(cur_state, cur_input, self.dt)  
        next_state_raw = next_state

        # Check whether next state is within the state space
        next_pos = max(min(next_state[0], self.pos_ubs), self.pos_lbs)
        next_vel = max(min(next_state[1], self.vel_ubs), self.vel_lbs)
        next_state = np.array([next_pos, next_vel])
        
        # Get reward
        next_state_index = self.nearest_state_index_lookup(next_state)
        if np.all(self.state_space[:, next_state_index] == self.env.target_state):
            reward = 10
        elif np.any(next_state_raw != next_state):
            reward = -10
        else:
            reward = -1
        
        return next_state, reward
    
    def nearest_state_index_lookup(self, state):
        """
        Find the nearest state index in the discrete state space for a given state.
        """
        distances = np.linalg.norm(self.state_space.T - np.array(state), axis=1)
        return np.argmin(distances)
    
    def build_stochastic_mdp(self):
        """
        Construct transition probability (T) and reward (R) matrices for the MDP.
        """
        # Unique values for position and velocity
        unique_pos = self.pos_partitions
        unique_vel = self.vel_partitions

        # Iterate over all states
        for state_index in range(self.num_states):
            cur_state = self.state_space[:, state_index]
            print(f"Building model... state {state_index + 1}/{self.num_states}")

            # Apply each possible action
            for action_index in range(self.num_actions):
                action = self.input_space[action_index]

                print(f"cur_state: {cur_state}")
                print(f"action: {action}")

                # Propagate forward to get next state and reward
                next_state, reward = self.one_step_forward(cur_state, action)

                print(f"next_state: {next_state}")

                # Find the two closest discretized values for position and velocity
                pos_indices = np.argsort(np.abs(unique_pos - next_state[0]))[:2]
                vel_indices = np.argsort(np.abs(unique_vel - next_state[1]))[:2]

                pos_bounds = [unique_pos[pos_indices[0]], unique_pos[pos_indices[1]]]
                vel_bounds = [unique_vel[vel_indices[0]], unique_vel[vel_indices[1]]]

                print(f"pos_bounds: {pos_bounds}")
                print(f"vel_bounds: {vel_bounds}")

                # Normalize next state within bounds
                x_norm = (next_state[0] - min(pos_bounds)) / (max(pos_bounds) - min(pos_bounds))
                y_norm = (next_state[1] - min(vel_bounds)) / (max(vel_bounds) - min(vel_bounds))

                print(f"x_norm: {x_norm}")
                print(f"y_norm: {y_norm}")

                # Calculate bilinear interpolation probabilities
                probs = [
                    (1 - x_norm) * (1 - y_norm),  # bottom-left
                    x_norm * (1 - y_norm),        # bottom-right
                    x_norm * y_norm,              # top-right
                    (1 - x_norm) * y_norm         # top-left
                ]

                # Four vertices of the enclosing box
                nodes = [
                    [min(pos_bounds), min(vel_bounds)],  # bottom-left
                    [max(pos_bounds), min(vel_bounds)],  # bottom-right
                    [max(pos_bounds), max(vel_bounds)],  # top-right
                    [min(pos_bounds), max(vel_bounds)]   # top-left
                ]

                # Initialize T and R matrices if not already done
                if self.T[action_index] is None:
                    self.T[action_index] = np.zeros((self.num_states, self.num_states))
                if self.R[action_index] is None:
                    self.R[action_index] = np.zeros((self.num_states, self.num_states))

                # Update transition and reward matrices
                for i, node in enumerate(nodes):
                    node_index = self.nearest_state_index_lookup(node)
                    self.T[action_index][state_index, node_index] += probs[i]
                    self.R[action_index][state_index, node_index] += reward

    # plot t for all states
    def plot_t(self):

        fig, ax = plt.subplots(2, self.num_actions, figsize=(15, 5))

        for i in range(self.num_actions):
            ax[0, i].imshow(self.T[i], cmap='hot', interpolation='nearest')
            ax[0, i].set_title(f"Action {i}")
            ax[0, i].set_xlabel("Next State")
            ax[0, i].set_ylabel("Current State")
        
        for i in range(self.num_actions):
            ax[1, i].imshow(self.R[i], cmap='hot', interpolation='nearest')
            ax[1, i].set_title(f"Action {i}")
            ax[1, i].set_xlabel("Next State")
            ax[1, i].set_ylabel("Current State")

        plt.tight_layout()
        plt.show()

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

def check_input_constraints(compute_action, mode='clipping'):
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
                if mode == 'raise_error':
                    raise ValueError(f"Warning: raw control input u={u} is beyond the lower limit {input_lbs}!")
                elif mode == 'clipping':
                    print(f"Warning: raw control input u={u} is beyond the lower limit {input_lbs}! Clip to lower limit u={input_lbs}.")
                    u = np.array([input_lbs])

        if input_ubs is not None:
            if not np.all(u <= input_ubs):
                if mode == 'raise_error':
                    raise ValueError(f"Warning: control input u={u} is beyond the upper limit {input_ubs}!")
                elif mode == 'clipping':
                    print(f"Warning: raw control input u={u} is beyond the upper limit {input_ubs}! Clip to upper limit u={input_ubs}.")
                    u = np.array([input_ubs])

        return u
    
    return wrapper


class BaseController(ABC):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            freq: float, 
            name: str, 
            type: str,
            verbose: bool = False
        ) -> None:

        self.name = name
        self.type = type

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


'''
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
            Horizon: int,
            name: str = 'DP',
            type: str = 'DP', 
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
        #ocp.solver_options.globalization = 'FUNNEL_L1PEN_LINESEARCH'
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
'''


class DPController(BaseController):
    def __init__(
            self,
            env: Env,
            dynamics: Dynamics,
            Q: np.ndarray,
            R: np.ndarray,
            Qf: np.ndarray, 
            freq: float,
            Horizon: int,
            name: str = 'DP',
            type: str = 'DP', 
            verbose: bool = False
        ) -> None:

        super().__init__(env, dynamics, freq, name, verbose)

        self.Q = Q
        self.R = R
        self.Qf = Qf

        self.N = Horizon

        self.Ad, self.Bd = self.dynamics.get_linearized_AB_discrete(self.init_state, 0, self.dt)

        if self.env.state_lbs is None or self.env.state_ubs is None:
            raise ValueError("Constraints on states must been fully specified for Approximate Dynamic Programming!")
        
        # Number of grid points for discretization
        self.num_x1 = 60
        self.num_x2 = 40
        
        # Initialize state and input space
        self.X1 = None
        self.X2 = None
        
        # Initialize policy and cost-to-go
        self.U = None
        self.J = None
    
        self.setup()

    def terminal_cost(self, x):

        return x.T @ self.Qf @ x

    def stage_cost(self, x, u):

        return x.T @ self.Q @ x + self.R * u**2

    def recursion_cost(self, u, x):

        x_next = self.Ad @ x + self.Bd.flatten() * u

        # Use linear interpolation to find the value of J at the next state
        J_next = self.J_func(x_next[0], x_next[1])

        return self.stage_cost(x, u) + J_next

    def state_constraints(self, u, x):

        # Calculate the next state
        x_next = self.Ad @ x + self.Bd.flatten() * u

        # Extract the position and velocity from current and next state
        p_curr, v_curr = x[0], x[1]
        p_next, v_next = x_next[0], x_next[1]

        # Define constraints on current and next state
        return [
            p_curr - self.env.state_lbs[0],  # p_curr >= state_lbs[0]
            self.env.state_ubs[0] - p_curr,  # p_curr <= state_ubs[0]
            v_curr - self.env.state_lbs[1],  # v_curr >= state_lbs[1]
            self.env.state_ubs[1] - v_curr,   # v_curr <= state_ubs[1]
            p_next - self.env.state_lbs[0],  # p_next >= state_lbs[0]
            self.env.state_ubs[0] - p_next,  # p_next <= state_ubs[0]
            v_next - self.env.state_lbs[1],  # v_next >= state_lbs[1]
            self.env.state_ubs[1] - v_next   # v_next <= state_ubs[1]
        ]

    def setup(self) -> None:
        
        # Generate grid mesh for state space
        self.X1 = np.linspace(self.env.state_lbs[0]-self.target_state[0], self.env.state_ubs[0]-self.target_state[0], self.num_x1)
        self.X2 = np.linspace(self.env.state_lbs[1]-self.target_state[1], self.env.state_ubs[1]-self.target_state[1], self.num_x2)
        
        # Initialize policy and cost-to-go function, and generate interpolation over grid mesh
        self.U = np.zeros((self.num_x1, self.num_x2))
        self.U_func = interp2d(self.X1, self.X2, self.U.T, kind='linear')
        self.J = np.zeros((self.num_x1, self.num_x2))
        for i in range(self.num_x1): # Initialize cost matirx with terminal cost
            for j in range(self.num_x2):
                self.J[i, j] = self.terminal_cost(np.array([self.X1[i], self.X2[j]]))
        self.J_func = interp2d(self.X1, self.X2, self.J.T, kind='linear')
        
        # Iterate over time steps from backward to forward
        for n in range(self.N, 0, -1):
            J_temp = np.zeros((self.num_x1, self.num_x2))

            for i in range(self.num_x1):
                for j in range(self.num_x2):

                    # Extract current state
                    x_current = np.array([self.X1[i], self.X2[j]])

                    # Solve for optimal control input
                    res = minimize(
                        self.recursion_cost,
                        x0=0,
                        args=(x_current, ),
                        bounds=[(self.env.input_lbs, self.env.input_ubs)],
                        method='SLSQP',
                        constraints={
                            'type': 'ineq',
                            'fun': self.state_constraints,
                            'args': (x_current,)
                        }
                    )
                    
                    # Extract optimal policy and cost-to-go 
                    self.U[i, j] = res.x
                    J_temp[i, j] = res.fun
                    
                    if self.verbose:
                        print(f"Current state: {x_current}, Current time step: {n}")
                        print(f"Optimal control input: {self.U[i, j]}")
                        print(f"Optimal cost-to-go function: {J_temp[i, j]}")

            self.J = J_temp

            # Re-interpolation over grid mesh
            self.J_func = interp2d(self.X1, self.X2, self.J.T, kind='linear')
            self.U_func = interp2d(self.X1, self.X2, self.U.T, kind='linear')
        
        if self.verbose:
            print(f"Dynamic Programming policy with input constraints computed.")
            print(f"Optimal control input: {self.U}")
            print(f"Optimal cost-to-go function: {self.J}")

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_time: int) -> np.ndarray:

        if np.all(self.U == 0):
            raise ValueError("DP Policy is not computed. Call setup() first.")
        
        det_x = current_state - self.target_state
        
        # Use linear interpolation to find the value of U at the current state
        miu = self.U_func(det_x[0], det_x[1])

        return miu

    def plot_policy_and_cost(self):
        """
        Generates a plot with two subplots showing the policy (U) and cost-to-go (J)
        defined on a given grid space.
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Plot policy (U)
        im1 = axs[0].imshow(self.U, extent=[self.X1[0], self.X1[-1], self.X2[0], self.X2[-1]], origin='lower', aspect='auto', cmap='viridis')
        axs[0].set_title('Policy (U)')
        axs[0].set_xlabel('Car Position')
        axs[0].set_ylabel('Car Velocity')
        fig.colorbar(im1, ax=axs[0], orientation='vertical')

        #axs[0].set_xticks(self.X1[::grid_interval], minor=False)
        #axs[0].set_yticks(self.X2[::grid_interval], minor=False)
        #axs[0].grid(color='black', linestyle='--', linewidth=0.5, alpha=0.7)

        # Plot cost-to-go (J)
        im2 = axs[1].imshow(self.J, extent=[self.X1[0], self.X1[-1], self.X2[0], self.X2[-1]], origin='lower', aspect='auto', cmap='viridis')
        axs[1].set_title('Cost-to-Go (J)')
        axs[1].set_xlabel('Car Position')
        axs[1].set_ylabel('Car Velocity')
        fig.colorbar(im2, ax=axs[1], orientation='vertical')

        #axs[1].set_xticks(self.X1[::grid_interval], minor=False)
        #axs[1].set_yticks(self.X2[::grid_interval], minor=False)
        #axs[1].grid(color='black', linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        plt.show()


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
            type: str = 'LQR', 
            verbose: bool = True
        ) -> None:

        super().__init__(env, dynamics, freq, name, type, verbose)
        
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
        if self.type in ['LQR', 'MPC']:
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
        input_eq = np.zeros(self.dim_inputs)
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
            type: str = 'iLQR', 
            max_iter: int = 30, 
            tol: float = 2e-2, 
            verbose: bool = True
        ) -> None:

        self.Qf = Qf  # Terminal cost matrix

        self.max_iter = max_iter
        self.tol = tol

        self.K_k_arr = None
        self.u_kff_arr = None

        super().__init__(env, dynamics, Q, R, freq, name, type, verbose)

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
            type: str = 'MPC', 
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

        super().__init__(env, dynamics, Q, R, freq, name, type, verbose)

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


# Derived class for RL Controller, solved by General Policy Iteration
class GPIController(BaseController):
    def __init__(self, 
                 mdp: Env_rl, 
                 freq: float, 
                 gamma: float = 0.95,
                 precision_pe: float = 1e-6,
                 precision_pi: float = 1e-6,
                 max_ite_pe: int = 50,
                 max_ite_pi: int = 100,
                 name: str = 'GPI', 
                 type: str = 'GPI', 
                 verbose: bool = True
                 ) -> None:
        
        super().__init__(mdp, mdp.dynamics, freq, name, type, verbose)
        
        # VI: max_ite_pi = 1, max_ite_pe > 500, precision_pi not needed, reasonable precision_pe (1e-6)
        # PI: max_ite_pi > 50, max_ite_pe > 100, reasonable precision_pi (1e-6), precision_pe not needed
        self.precision_pe = precision_pe
        self.precision_pi = precision_pi
        self.max_ite_pe = max_ite_pe
        self.max_ite_pi = max_ite_pi

        self.mdp = mdp
        self.gamma = gamma

        self.dim_states = self.mdp.num_states
        self.dim_inputs = self.mdp.num_actions

        self.init_state = self.env.init_state
        self.target_state = self.env.target_state

        # Initialize policy and value function
        self.policy = np.zeros(self.dim_states, dtype=int)  # Initial policy: choose action 0 for all states
        self.value_function = np.zeros(self.dim_states)  # Initial value function
    
    def setup(self) -> None:
        """Perform GPI to compute the optimal policy."""

        new_value_function = np.zeros_like(self.value_function)

        for pi_iteration in range(self.max_ite_pi):

            # Policy Evaluation
            for pe_iteration in range(self.max_ite_pe):

                self.value_function = copy.copy(new_value_function)

                new_value_function = np.zeros_like(self.value_function)
                
                for state_index in range(self.dim_states):

                    action_index = self.policy[state_index]

                    for next_state_index in range(self.dim_states):

                        new_value_function[state_index] += self.mdp.T[action_index][state_index, next_state_index] * (
                            self.mdp.R[action_index][state_index, next_state_index] + self.gamma * self.value_function[next_state_index]
                        )

                print(f"self.value_function: {self.value_function}")
                print(f"new_value_function: {new_value_function}")

                # Check for convergence in policy evaluation
                if np.max(np.abs(new_value_function - self.value_function)) < self.precision_pe:
                    if self.verbose:
                        print(f"Policy evaluation converged after {pe_iteration + 1} iterations, max error: {np.max(np.abs(new_value_function - self.value_function))}.")
                    break
                elif self.verbose:
                    print(f"Policy evaluation iteration {pe_iteration + 1}, max error: {np.max(np.abs(new_value_function - self.value_function))}")
            
            self.value_function = copy.copy(new_value_function)
            
            # Policy Improvement
            policy_stable = True
            old_policy = copy.copy(self.policy)
            for state_index in range(self.dim_states):
                
                q_values = np.zeros(self.dim_inputs)

                # Compute Q-values for all actions
                for action_index in range(self.dim_inputs):
                    for next_state_index in range(self.dim_states):

                        q_values[action_index] += self.mdp.T[action_index][state_index, next_state_index] * (
                            self.mdp.R[action_index][state_index, next_state_index] + self.gamma * self.value_function[next_state_index]
                        )

                # Update policy greedily
                self.policy[state_index] = np.argmax(q_values)

                # Check for convergence in policy improvement
                if old_policy[state_index] != self.policy[state_index]:
                    policy_stable = False

            # Check for convergence in policy improvement
            if policy_stable and np.max(np.abs(new_value_function - self.value_function)) < self.precision_pi:
                if self.verbose:
                    print(f"Policy converged after {pi_iteration + 1} iterations.")
                    print(f"Optimal Policy: {self.policy}")
                break
            elif self.verbose:
                print(f"Policy improvement iteration {pi_iteration + 1}, still not converged, keep running.")
                print(f"Last Policy: {old_policy}")
                print(f"Current Policy: {self.policy}")

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_time) -> np.ndarray:
        """
        Use the optimal policy to compute the action for the given state.
        """
        # Find the nearest discrete state index
        state_index = self.mdp.nearest_state_index_lookup(current_state)
        
        # Get the optimal action from the policy
        action_index = self.policy[state_index]
        action = self.mdp.input_space[action_index]

        return np.array([action])

    def plot_heatmaps(self):
        """
        Visualize the policy and cost as a 2D state-value map.
        """
        value_map = self.value_function.reshape(self.mdp.num_pos, self.mdp.num_vel)
        policy_map = self.policy.reshape(self.mdp.num_pos, self.mdp.num_vel)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Plot policy (U)
        im1 = axs[0].imshow(policy_map, extent=[
            self.mdp.pos_partitions[0], self.mdp.pos_partitions[-1],
            self.mdp.vel_partitions[0], self.mdp.vel_partitions[-1]
        ], origin='lower', aspect='auto', cmap='viridis')
        axs[0].set_title('Optimal Policy')
        axs[0].set_xlabel('Car Position')
        axs[0].set_ylabel('Car Velocity')
        fig.colorbar(im1, ax=axs[0], orientation='vertical')

        # Plot cost-to-go (J)
        im2 = axs[1].imshow(value_map, extent=[
            self.mdp.pos_partitions[0], self.mdp.pos_partitions[-1],
            self.mdp.vel_partitions[0], self.mdp.vel_partitions[-1]
        ], origin='lower', aspect='auto', cmap='viridis')
        axs[1].set_title('Optimal Cost')
        axs[1].set_xlabel('Car Position')
        axs[1].set_ylabel('Car Velocity')
        fig.colorbar(im2, ax=axs[1], orientation='vertical')

        plt.tight_layout()
        plt.show()



# Derived class for RL Controller, solved by Q-Learning
class QLearningController(BaseController):
    def __init__(self, 
                 mdp: Env_rl, 
                 freq: float, 
                 epsilon: float = 0.3, 
                 k_epsilon: float = 0.99, 
                 epsilon_min: float = 0.01,
                 learning_rate: float = 0.2, 
                 gamma: float = 0.95,
                 max_iterations: int = 1000, 
                 max_steps_per_episode: int = 100, 
                 name: str = 'Q-Learning', 
                 type: str = 'Q-Learning', 
                 verbose: bool = True
                 ) -> None:
        
        super().__init__(mdp, mdp.dynamics, freq, name, type, verbose)

        self.epsilon = epsilon  # exploration rate
        self.k_epsilon = k_epsilon  # decay factor for exploration rate
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.max_steps_per_episode = max_steps_per_episode

        self.mdp = mdp

        self.dim_states = self.mdp.num_states
        self.dim_inputs = self.mdp.num_actions

        self.init_state = self.env.init_state
        self.target_state = self.env.target_state

        # Initialize Q table with all 0s
        self.Q = np.zeros((self.dim_states, self.dim_inputs)) 
        
        # Initialize policy as None
        self.policy = np.zeros((self.dim_states)) 
        self.value_function = np.zeros((self.dim_states)) 

        # For training curve plotting
        self.residual_rewards = []
        self.epsilon_list = []
        self.SR_100epoch = [] # Successful rounds
        self.F_100epoch = [] # Failure rounds
        self.TO_100epoch = [] # Time out rounds

    def _get_action_probabilities(self, state_index: int) -> np.ndarray:
        """Calculate the action probabilities using epsilon-soft policy."""

        probabilities = np.ones(self.dim_inputs) * (self.epsilon / self.dim_inputs)
        best_action = np.argmax(self.Q[state_index])
        probabilities[best_action] += (1.0 - self.epsilon)

        return probabilities

    def setup(self) -> None:

        for iteration in range(self.max_iterations):

            total_reward = 0  # To accumulate rewards for this episode

            if iteration % 100 == 0:

                if iteration != 0:
                    # Record the SR_100epoch, F_100epoch, TO_100epoch
                    self.SR_100epoch.append(SR_100epoch)
                    self.F_100epoch.append(F_100epoch)
                    self.TO_100epoch.append(TO_100epoch)
                
                SR_100epoch = 0
                F_100epoch = 0
                TO_100epoch = 0

            # Ramdomly choose a state to start
            current_state_index = np.random.choice(self.dim_states)
            current_state = self.mdp.state_space[:, current_state_index]

            for step in range(self.max_steps_per_episode):

                # Choose action based on epsilon-soft policy
                action_probabilities = self._get_action_probabilities(current_state_index)
                action_index = np.random.choice(np.arange(self.dim_inputs), p=action_probabilities)
                current_input = self.mdp.input_space[action_index]

                # Take action and observe the next state and reward
                next_state, reward = self.mdp.one_step_forward(current_state, current_input)
                next_state_index = self.mdp.nearest_state_index_lookup(next_state)
                total_reward += reward  # Accumulate total reward
                
                if self.verbose:
                    print(f"reward: {reward}")
                    print(f"total_reward: {total_reward}")
                    print(f"current_state: {current_state}, current_input: {current_input}, next_state: {next_state}, next_state_render: {self.mdp.state_space[:, next_state_index]}, reward: {reward}")

                # Check if the episode is finished
                terminate_condition_1 = False#next_state[0]==self.mdp.pos_partitions[-1]
                terminate_condition_2 = False#next_state[0]==self.mdp.pos_partitions[0]
                terminate_condition_3 = np.all(self.mdp.state_space[:, next_state_index]==self.target_state)

                if terminate_condition_1 or terminate_condition_2 or terminate_condition_3:

                    if terminate_condition_3:
                        SR_100epoch += 1
                    else:
                        F_100epoch += 1

                    if self.verbose:
                        print(f"Iteration {iteration + 1}/{self.max_iterations}: finished successfully! epsilon: {self.epsilon:.4f}, residual reward: {total_reward:.2f}")
                    
                    break

                else:
                    # Update Q table
                    q_update = reward + self.gamma * np.max(self.Q[next_state_index, :])
                    self.Q[current_state_index, action_index] += self.learning_rate * (
                        q_update - self.Q[current_state_index, action_index]
                    )
                    
                    # Move to the next state
                    current_state_index = next_state_index
                    current_state = self.mdp.state_space[:, current_state_index]
                
                if step == self.max_steps_per_episode-1:
                    TO_100epoch +=1
                    if self.verbose:
                        print(f"Iteration {iteration + 1}/{self.max_iterations}: time out! epsilon: {self.epsilon:.4f}, residual reward: {total_reward:.2f}")
                    

            # Decrease epsilon
            self.epsilon *= self.k_epsilon
            self.epsilon = max(self.epsilon, self.epsilon_min)
            self.epsilon_list.append(self.epsilon)

            # Record the residual reward
            self.residual_rewards.append(total_reward)
        
        # Return the deterministic policy and value function
        self.policy = np.argmax(self.Q, axis=1)
        self.value_function = np.max(self.Q, axis=1)
        
        if self.verbose:
            print("Training finished！")

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_time) -> np.ndarray:
        """
        Use the optimal policy to compute the action for the given state.
        """
        # Find the nearest discrete state index
        state_index = self.mdp.nearest_state_index_lookup(current_state)
        
        # Get the optimal action from the policy
        action_index = self.policy[state_index]
        action = self.mdp.input_space[action_index]

        return np.array([action])

    def plot_heatmaps(self):
        """
        Visualize the policy and cost as a 2D state-value map.
        """
        value_map = self.value_function.reshape(self.mdp.num_pos, self.mdp.num_vel)
        policy_map = self.policy.reshape(self.mdp.num_pos, self.mdp.num_vel)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Plot policy (U)
        im1 = axs[0].imshow(policy_map, extent=[
            self.mdp.pos_partitions[0], self.mdp.pos_partitions[-1],
            self.mdp.vel_partitions[0], self.mdp.vel_partitions[-1]
        ], origin='lower', aspect='auto', cmap='viridis')
        axs[0].set_title('Optimal Policy')
        axs[0].set_xlabel('Car Position')
        axs[0].set_ylabel('Car Velocity')
        fig.colorbar(im1, ax=axs[0], orientation='vertical')

        # Plot cost-to-go (J)
        im2 = axs[1].imshow(value_map, extent=[
            self.mdp.pos_partitions[0], self.mdp.pos_partitions[-1],
            self.mdp.vel_partitions[0], self.mdp.vel_partitions[-1]
        ], origin='lower', aspect='auto', cmap='viridis')
        axs[1].set_title('Optimal Cost')
        axs[1].set_xlabel('Car Position')
        axs[1].set_ylabel('Car Velocity')
        fig.colorbar(im2, ax=axs[1], orientation='vertical')

        plt.tight_layout()
        plt.show()

    def plot_training_curve(self):
        """
        Visualize the training curve of residual rewards and holdsignal for sr_100epoch in 2 figure.
        """

        self.SR_100epoch = np.repeat(self.SR_100epoch, 100)
        self.F_100epoch = np.repeat(self.F_100epoch, 100)
        self.TO_100epoch = np.repeat(self.TO_100epoch, 100)
        
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))

        # Plot residual rewards
        axs[0].plot(self.residual_rewards)
        axs[0].set_title('Total Rewards')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Total Reward')

        # Plot epsilon
        axs[1].plot(self.epsilon_list)
        axs[1].set_title('Epsilon')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Epsilon')
        axs[1].set_ylim(0, 1)

        # Plot SR_100epoch, F_100epoch, TO_100epoch
        axs[2].plot(self.SR_100epoch, label='Success rounds / 100Epoch')
        axs[2].plot(self.F_100epoch, label='Fail rounds / 100Epoch')
        axs[2].plot(self.TO_100epoch, label='Time out / 100Epoch')
        axs[2].set_title('Statictics / 100 Epoch')
        axs[2].set_xlabel('Iteration')
        axs[2].set_ylabel('Number of rounds')
        axs[2].set_ylim(0, 100)
        axs[2].legend()


        plt.tight_layout()
        plt.show()


# Derived class for RL Controller, solved by MCRL
class MCRLController(BaseController):
    def __init__(self, 
                 mdp: Env_rl, 
                 freq: float, 
                 epsilon: float = 0.3, 
                 k_epsilon: float = 0.99, 
                 epsilon_min: float = 0.01,
                 learning_rate: float = 0.2, 
                 gamma: float = 0.95,
                 max_iterations: int = 1000, 
                 max_steps_per_episode: int = 100, 
                 name: str = 'MCRL', 
                 type: str = 'MCRL', 
                 verbose: bool = True
                 ) -> None:
        
        super().__init__(mdp, mdp.dynamics, freq, name, type, verbose)

        self.epsilon = epsilon  # exploration rate
        self.k_epsilon = k_epsilon  # decay factor for exploration rate
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.max_steps_per_episode = max_steps_per_episode

        self.mode = "every_visit"  # "first_visit" or "every_visit"

        self.mdp = mdp

        self.dim_states = self.mdp.num_states
        self.dim_inputs = self.mdp.num_actions

        self.init_state = self.env.init_state
        self.target_state = self.env.target_state

        # Initialize Q table with all 0s
        self.Q = np.zeros((self.dim_states, self.dim_inputs)) 
        self.state_action_counts = np.zeros((self.dim_states, self.dim_inputs))
        
        # Initialize policy as None
        self.policy = np.zeros((self.dim_states)) 
        self.value_function = np.zeros((self.dim_states)) 

        # For training curve plotting
        self.residual_rewards = []
        self.epsilon_list = []
        self.SR_100epoch = [] # Successful rounds
        self.F_100epoch = [] # Failure rounds
        self.TO_100epoch = [] # Time out rounds

    def _get_action_probabilities(self, state_index: int) -> np.ndarray:
        """Calculate the action probabilities using epsilon-soft policy."""

        probabilities = np.ones(self.dim_inputs) * (self.epsilon / self.dim_inputs)
        best_action = np.argmax(self.Q[state_index])
        probabilities[best_action] += (1.0 - self.epsilon)

        return probabilities

    def setup(self) -> None:

        for iteration in range(self.max_iterations):

            episode = []  # storage state, action and reward for current episode
            total_reward = 0  # total reward for current episode

            if iteration % 100 == 0:

                if iteration != 0:
                    # Record the SR_100epoch, F_100epoch, TO_100epoch
                    self.SR_100epoch.append(SR_100epoch)
                    self.F_100epoch.append(F_100epoch)
                    self.TO_100epoch.append(TO_100epoch)
                
                SR_100epoch = 0
                F_100epoch = 0
                TO_100epoch = 0

            # Randomly choose a state to start
            current_state_index = np.random.choice(self.dim_states)
            current_state = self.mdp.state_space[:, current_state_index]
            
            # Generate an episode
            for step in range(self.max_steps_per_episode):
                # Choose action based on epsilon-soft policy
                action_probabilities = self._get_action_probabilities(current_state_index)
                action_index = np.random.choice(np.arange(self.dim_inputs), p=action_probabilities)
                current_input = self.mdp.input_space[action_index]

                # Take action and observe the next state and reward
                next_state, reward = self.mdp.one_step_forward(current_state, current_input)
                next_state_index = self.mdp.nearest_state_index_lookup(next_state)
                total_reward += reward

                # Store the state, action and reward for this step
                episode.append((current_state_index, action_index, reward))

                # Check if the episode is finished
                terminate_condition_1 = False#next_state[0]==self.mdp.pos_partitions[-1]
                terminate_condition_2 = False#next_state[0]==self.mdp.pos_partitions[0]
                terminate_condition_3 = np.all(self.mdp.state_space[:, next_state_index]==self.target_state)

                if terminate_condition_1 or terminate_condition_2 or terminate_condition_3:

                    if terminate_condition_3:
                        SR_100epoch += 1
                    else:
                        F_100epoch += 1

                    if self.verbose:
                        print(f"Iteration {iteration + 1}/{self.max_iterations}: finished successfully! epsilon: {self.epsilon:.4f}, residual reward: {total_reward:.2f}")
                    
                    break

                if step == self.max_steps_per_episode-1:
                    TO_100epoch +=1
                    if self.verbose:
                        print(f"Iteration {iteration + 1}/{self.max_iterations}: time out! epsilon: {self.epsilon:.4f}, residual reward: {total_reward:.2f}")
                    
                # Move to the next state
                current_state_index = next_state_index
                current_state = self.mdp.state_space[:, current_state_index]

            # Update Q table using Monte Carlo method
            G = 0  # Return
            if self.mode == "first_visit":
                visited = set()

            for state_index, action_index, reward in reversed(episode):
                G = reward + self.gamma * G
                
                 # update Q table
                if self.mode == "first_visit" and (state_index, action_index) not in visited:
                    visited.add((state_index, action_index))
                    self.state_action_counts[state_index, action_index] += 1
                    alpha = 1.0 / self.state_action_counts[state_index, action_index]
                    self.Q[state_index, action_index] += alpha * (G - self.Q[state_index, action_index])

                elif self.mode == "every_visit":
                    self.state_action_counts[state_index, action_index] += 1
                    alpha = 1.0 / self.state_action_counts[state_index, action_index]
                    self.Q[state_index, action_index] += alpha * (G - self.Q[state_index, action_index])

            # Decrease epsilon
            self.epsilon *= self.k_epsilon
            self.epsilon = max(self.epsilon, self.epsilon_min)
            self.epsilon_list.append(self.epsilon)

            # Record the residual reward
            self.residual_rewards.append(total_reward)

        # Return the deterministic policy and value function
        self.policy = np.argmax(self.Q, axis=1)
        self.value_function = np.max(self.Q, axis=1)

        if self.verbose:
            print("Training finished！")

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_time) -> np.ndarray:
        """
        Use the optimal policy to compute the action for the given state.
        """
        # Find the nearest discrete state index
        state_index = self.mdp.nearest_state_index_lookup(current_state)
        
        # Get the optimal action from the policy
        action_index = self.policy[state_index]
        action = self.mdp.input_space[action_index]

        return np.array([action])

    def plot_heatmaps(self):
        """
        Visualize the policy and cost as a 2D state-value map.
        """
        value_map = self.value_function.reshape(self.mdp.num_pos, self.mdp.num_vel)
        policy_map = self.policy.reshape(self.mdp.num_pos, self.mdp.num_vel)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        # Plot policy (U)
        im1 = axs[0].imshow(policy_map, extent=[
            self.mdp.pos_partitions[0], self.mdp.pos_partitions[-1],
            self.mdp.vel_partitions[0], self.mdp.vel_partitions[-1]
        ], origin='lower', aspect='auto', cmap='viridis')
        axs[0].set_title('Optimal Policy')
        axs[0].set_xlabel('Car Position')
        axs[0].set_ylabel('Car Velocity')
        fig.colorbar(im1, ax=axs[0], orientation='vertical')

        # Plot cost-to-go (J)
        im2 = axs[1].imshow(value_map, extent=[
            self.mdp.pos_partitions[0], self.mdp.pos_partitions[-1],
            self.mdp.vel_partitions[0], self.mdp.vel_partitions[-1]
        ], origin='lower', aspect='auto', cmap='viridis')
        axs[1].set_title('Optimal Cost')
        axs[1].set_xlabel('Car Position')
        axs[1].set_ylabel('Car Velocity')
        fig.colorbar(im2, ax=axs[1], orientation='vertical')

        plt.tight_layout()
        plt.show()

    def plot_training_curve(self):
        """
        Visualize the training curve of residual rewards and holdsignal for sr_100epoch in 2 figure.
        """

        self.SR_100epoch = np.repeat(self.SR_100epoch, 100)
        self.F_100epoch = np.repeat(self.F_100epoch, 100)
        self.TO_100epoch = np.repeat(self.TO_100epoch, 100)
        
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))

        # Plot residual rewards
        axs[0].plot(self.residual_rewards)
        axs[0].set_title('Total Rewards')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Total Reward')

        # Plot epsilon
        axs[1].plot(self.epsilon_list)
        axs[1].set_title('Epsilon')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Epsilon')
        axs[1].set_ylim(0, 1)

        # Plot SR_100epoch, F_100epoch, TO_100epoch
        axs[2].plot(self.SR_100epoch, label='Success rounds / 100Epoch')
        axs[2].plot(self.F_100epoch, label='Fail rounds / 100Epoch')
        axs[2].plot(self.TO_100epoch, label='Time out / 100Epoch')
        axs[2].set_title('Statictics / 100 Epoch')
        axs[2].set_xlabel('Iteration')
        axs[2].set_ylabel('Number of rounds')
        axs[2].set_ylim(0, 100)
        axs[2].legend()


        plt.tight_layout()
        plt.show()










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

        # Define setting of plots and animation
        self.color = "blue"
        self.color_list = ['red', 'green', 'yellow', 'orange']
        self.car_length= 0.2
        self.figsize = (8, 4)
        self.refresh_rate = 30

    def display_plots(self) -> None:

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))

        # Plot p over time t

        ax1.plot(self.t_eval, self.position, label="Position p(t)", color="blue")

        if self.env.state_lbs is not None:
            ax1.fill_between(self.t_eval, ax1.get_ylim()[0], self.env.state_lbs[0], facecolor='gray', alpha=0.3, label=f'pos_lower_bound')
        if self.env.state_ubs is not None:
            ax1.fill_between(self.t_eval, self.env.state_ubs[0], ax1.get_ylim()[1], facecolor='gray', alpha=0.3, label=f'pos_upper_bound')

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.legend()

        # Plot v over time t
        ax2.plot(self.t_eval, self.velocity, label="Velocity v(t)", color="green")

        if self.env.state_lbs is not None:
            ax2.fill_between(self.t_eval, ax2.get_ylim()[0], self.env.state_lbs[1], facecolor='gray', alpha=0.3, label=f'vel_lower_bound')
        if self.env.state_ubs is not None:
            ax2.fill_between(self.t_eval, self.env.state_ubs[1], ax2.get_ylim()[1], facecolor='gray', alpha=0.3, label=f'vel_upper_bound')
            
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.legend()

        # Plot a over time t
        ax3.plot(self.t_eval, self.acceleration, label="Acceleration a(t)", color="red")

        if self.env.input_lbs is not None:
            ax3.fill_between(self.t_eval, ax3.get_ylim()[0], self.env.input_lbs, facecolor='gray', alpha=0.3, label=f'acc_lower_bound')
        if self.env.input_ubs is not None:
            ax3.fill_between(self.t_eval, self.env.input_ubs, ax3.get_ylim()[1], facecolor='gray', alpha=0.3, label=f'acc_upper_bound')

        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Acceleration (m/s^2)")
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def display_contrast_plots(self, *simulators: Simulator) -> None:

        color_index = 0

        if not simulators:
            raise ValueError("No simulator references provided.")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))

        # Plot the reference and evaluated trajectories for each simulator
        for simulator_ref in simulators:
            if not simulator_ref.state_traj:
                raise ValueError(f"Failed to get trajectory from simulator {simulator_ref.controller.name}. State trajectory list is void; please run 'run_simulation' first.")

            # Get reference trajectories from simulator_ref
            state_traj_ref, input_traj_ref = simulator_ref.get_trajectories()

            # Extract reference position, velocity, and acceleration
            position_ref = state_traj_ref[:, 0]
            velocity_ref = state_traj_ref[:, 1]
            acceleration_ref = input_traj_ref

            # Plot position over time
            ax1.plot(self.t_eval, position_ref, linestyle="--", label=f"{simulator_ref.controller.name} Position", color=self.color_list[color_index])

            # Plot velocity over time
            ax2.plot(self.t_eval, velocity_ref, linestyle="--", label=f"{simulator_ref.controller.name} Velocity", color=self.color_list[color_index])

            # Plot acceleration over time
            ax3.plot(self.t_eval, acceleration_ref, linestyle="--", label=f"{simulator_ref.controller.name} Acceleration", color=self.color_list[color_index])
            
            color_index += 1

        # Plot current object's trajectories
        ax1.plot(self.t_eval, self.position, label=f"{self.controller.name} Position", color=self.color)
        if self.env.state_lbs is not None:
            ax1.fill_between(self.t_eval, ax1.get_ylim()[0], self.env.state_lbs[0], facecolor='gray', alpha=0.3, label=f'pos_lower_bound')
        if self.env.state_ubs is not None:
            ax1.fill_between(self.t_eval, self.env.state_ubs[0], ax1.get_ylim()[1], facecolor='gray', alpha=0.3, label=f'pos_upper_bound')

        ax2.plot(self.t_eval, self.velocity, label=f"{self.controller.name} Velocity", color=self.color)
        if self.env.state_lbs is not None:
            ax2.fill_between(self.t_eval, ax2.get_ylim()[0], self.env.state_lbs[1], facecolor='gray', alpha=0.3, label=f'vel_lower_bound')
        if self.env.state_ubs is not None:
            ax2.fill_between(self.t_eval, self.env.state_ubs[1], ax2.get_ylim()[1], facecolor='gray', alpha=0.3, label=f'vel_upper_bound')

        ax3.plot(self.t_eval, self.acceleration, label=f"{self.controller.name} Acceleration", color=self.color)
        if self.env.input_lbs is not None:
            ax3.fill_between(self.t_eval, ax3.get_ylim()[0], self.env.input_lbs, facecolor='gray', alpha=0.3, label=f'acc_lower_bound')
        if self.env.input_ubs is not None:
            ax3.fill_between(self.t_eval, self.env.input_ubs, ax3.get_ylim()[1], facecolor='gray', alpha=0.3, label=f'acc_upper_bound')

        # Set labels and legends
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.legend()

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.legend()

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
    
    def display_contrast_animation(self, *simulators) -> HTML:

        # Instantiate the plotting
        num_plots = len(simulators) + 1
        fig, axes = plt.subplots(num_plots, 1, figsize=(self.figsize[0], self.figsize[1] * num_plots), sharex=True)

        # Define size of plotting
        p_max = max(max(sim.get_trajectories()[0][:, 0]) for sim in simulators)
        p_min = min(min(sim.get_trajectories()[0][:, 0]) for sim in simulators)
        start_extension = p_min - 0.3
        end_extension = p_max + 0.3

        p_disp_vals = np.linspace(start_extension, end_extension, 50)  # generate grid mesh on p

        h_disp_vals = [float(self.env.h(p).full().flatten()[0]) for p in p_disp_vals]
        h_max = max(h_disp_vals)
        h_min = min(h_disp_vals)
        axes[0].set_xlim(start_extension - 0.5, end_extension + 0.5)
        axes[0].set_ylim(h_min - 0.2, h_max + 0.3)
        axes[0].plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="green")
        axes[0].set_xlabel("Position p")
        axes[0].set_ylabel("Height h")

        for ax, sim in zip(axes[1:], simulators):

            h_disp_vals = [float(sim.env.h(p).full().flatten()[0]) for p in p_disp_vals]
            h_max = max(h_disp_vals)
            h_min = min(h_disp_vals)
            ax.set_xlim(start_extension - 0.5, end_extension + 0.5)
            ax.set_ylim(h_min - 0.2, h_max + 0.3)
            ax.plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="green")
            ax.set_xlabel("Position p")
            ax.set_ylabel("Height h")


        # Mark the initial state and the target state in the plotting
        initial_h = float(self.env.h(self.env.initial_position).full().flatten()[0])
        target_h = float(self.env.h(self.env.target_position).full().flatten()[0])
        axes[0].scatter([self.env.initial_position], [initial_h], color="blue", label="Initial position")
        axes[0].scatter([self.env.target_position], [target_h], color="orange", label="Target position")
        axes[0].legend()

        for ax, sim in zip(axes[1:], simulators):

            initial_h = float(sim.env.h(self.env.initial_position).full().flatten()[0])
            target_h = float(sim.env.h(self.env.target_position).full().flatten()[0])
            ax.scatter([sim.env.initial_position], [initial_h], color="blue", label="Initial position")
            ax.scatter([sim.env.target_position], [target_h], color="orange", label="Target position")
            ax.legend()


        # Create car objects for each simulator
        car_objects = {}
        colors = self.color_list[:len(simulators)]
        car_height = self.car_length / 2

        car_self = Rectangle((0, 0), self.car_length, car_height, color=self.color)
        axes[0].add_patch(car_self)
        axes[0].set_title(f"{self.controller.name}")

        for ax, sim, color in zip(axes[1:], simulators, colors):
            car = Rectangle((0, 0), self.car_length, car_height, color=color)
            ax.add_patch(car)
            ax.set_title(f"{sim.controller.name}")
            car_objects[sim] = car


        def update(frame):
            # Update car for self
            current_position = self.position[frame]
            current_theta = float(self.env.theta(current_position).full().flatten()[0])
            car_self.set_xy((current_position, float(self.env.h(current_position).full().flatten()[0])))
            car_self.angle = np.degrees(current_theta)  # rad to deg

            for sim, car in car_objects.items():
                current_position = sim.get_trajectories()[0][:, 0][frame]
                current_theta = float(sim.env.theta(current_position).full().flatten()[0])
                car.set_xy((current_position, float(sim.env.h(current_position).full().flatten()[0])))
                car.angle = np.degrees(current_theta)  # rad to deg

        # Instantiate animation
        anim = FuncAnimation(fig, update, frames=len(self.t_eval), interval=1000 / self.refresh_rate, repeat=False)

        return HTML(anim.to_jshtml())



# Actions:
# add a dataclass to set up all internal & public variable in class (https://docs.python.org/3/library/dataclasses.html)


# Next steps:
# finish PID
