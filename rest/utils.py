import numpy as np
import sympy as sp
import pickle
import casadi as ca
import scipy.linalg
import copy
import time
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Callable, Union, Optional, Tuple, Any
from functools import wraps
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from IPython.display import display, HTML
from scipy.interpolate import interp2d
from cvxopt import matrix, solvers
import pytope as pt
from scipy.spatial import ConvexHull, QhullError

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
            input_ubs: float = None,
            disturbance_lbs: np.ndarray = None, 
            disturbance_ubs: np.ndarray = None
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

        self.disturbance_lbs = disturbance_lbs
        self.disturbance_ubs = disturbance_ubs

        # Define argument p as CasADi symbolic parameters
        p = ca.SX.sym("p") # p: horizontal displacement

        # Initialize symbolic h and theta if not given
        if symbolic_h == None:

            def symbolic_h(p, case):

                if case == 1:  # zero slope
                    h = 0

                elif case == 2: # constant slope
                    h = (ca.pi * p) / 18

                elif case == 3: # varying slope (small disturbance)
                    h = 0.005 * ca.cos(18 * p)

                elif case == 4: # varying slope (underactated case)

                    condition_left = p <= -ca.pi/2
                    condition_right = p >= ca.pi/6
                    
                    h_center = ca.sin(3 * p)
                    h_flat = 1

                    h = ca.if_else(condition_left, h_flat, ca.if_else(condition_right, h_flat, h_center))
                    
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

        self.p_vals_disp = np.linspace(lbs_position, ubs_position, 200)
    
    def show_slope(self) -> None:
        
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
        ax[0].set_xlabel("p")
        ax[0].set_ylabel("h")
        ax[0].set_ylim(-1.4, 1.4)  
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
        ax[0].scatter([self.initial_position], [initial_h], color="blue", label="Start")
        #ax[0].scatter([self.target_position], [target_h], color="orange", label="Target")
        ax[0].plot(self.target_position, target_h, marker='x', color='red', markersize=10, markeredgewidth=3, label='Target')
        ax[0].set_xlabel("p")
        ax[0].set_ylabel("h")
        ax[0].set_ylim(-1.4, 1.4)  
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
            setup_dynamics: Callable[[ca.Function], Callable[[ca.SX, ca.SX], ca.SX]] = None
        ) -> None:

        # Initialize system dynmaics if not given
        if state_names is None and input_names is None:
            state_names = ["p", "v"]
            input_names = ["u"]

        # Define state and input as CasADi symbolic parameters
        self.states = ca.vertcat(*[ca.SX.sym(name) for name in state_names])
        self.inputs = ca.vertcat(*[ca.SX.sym(name) for name in input_names])

        self.dim_states = self.states.shape[0]
        self.dim_inputs = self.inputs.shape[0]

        self.env = env

        # Initialize system dynmaics if not given
        if setup_dynamics is None:

            def setup_dynamics(theta_function):

                p = ca.SX.sym("p")
                v = ca.SX.sym("v")
                u = ca.SX.sym("u")
                Gravity = 9.81

                theta = theta_function(p)
        
                # Expression of dynamics
                dpdt = v
                dvdt = u * ca.cos(theta) - Gravity * ca.sin(theta) * ca.cos(theta)
                
                state = ca.vertcat(p, v)
                input = ca.vertcat(u)
                rhs = ca.vertcat(dpdt, dvdt)

                return ca.Function("dynamics_function", [state, input], [rhs])
            
        # Define system dynamics
        self.dynamics_function = setup_dynamics(self.env.theta)

        # Initialize Jacobians
        self.A_c_func = None
        self.B_c_func = None

    def linearization(
        self,
        current_state: np.ndarray, 
        current_input: np.ndarray, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute symbolic Jacobians A(x,u) & B(x,u) and state / input matrix A & B of the system dynamics.
        """

        f = self.dynamics_function(self.states, self.inputs)
        A_c_sym = ca.jacobian(f, self.states)
        B_c_sym = ca.jacobian(f, self.inputs)

        self.A_c_func = ca.Function("A_func", [self.states, self.inputs], [A_c_sym])
        self.B_c_func = ca.Function("B_func", [self.states, self.inputs], [B_c_sym])

        A_c = np.array(self.A_c_func(current_state, current_input))
        B_c = np.array(self.B_c_func(current_state, current_input))

        return A_c, B_c
    
    def discretization(
        self,
        A_c: np.ndarray,
        B_c: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize continuous-time linear system x_dot = A_c x + B_c u
        using matrix exponential method (Zero-Order Hold).

        Returns:
            A_d, B_d: Discrete-time system matrices
        """

        # Construct augmented matrix
        aug_matrix = np.zeros((self.dim_states + self.dim_inputs, self.dim_states + self.dim_inputs))
        aug_matrix[:self.dim_states, :self.dim_states] = A_c
        aug_matrix[:self.dim_states, self.dim_states:] = B_c

        # Compute matrix exponential
        exp_matrix = scipy.linalg.expm(aug_matrix * dt)

        # Extract A_d and B_d
        A_d = exp_matrix[:self.dim_states, :self.dim_states]
        B_d = exp_matrix[:self.dim_states, self.dim_states:]

        return A_d, B_d

    def get_linearized_AB_discrete(
            self, 
            current_state: np.ndarray, 
            current_input: np.ndarray, 
            dt: float
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''use current state, and current input to calculate the linearized state transfer matrix A_lin and input matrix B_lin'''
        
        # Linearize the system dynamics
        A_c, B_c = self.linearization(current_state, current_input)

        # Discretize the system dynamics
        A_d, B_d = self.discretization(A_c, B_c, dt)
        
        # Check controllability of the discretized system
        controllability_matrix = np.hstack([np.linalg.matrix_power(A_d, i) @ B_d for i in range(self.dim_states)])
        rank = np.linalg.matrix_rank(controllability_matrix)

        if rank < self.dim_states:
            raise ValueError(f"System (A, B) is not controllable，rank of controllability matrix is {rank}, while the dimension of state space is {self.dim_states}.")
        
        return A_d, B_d
    
    def get_linearized_AB_discrete_sym(self, dt: float) -> Tuple[ca.Function, ca.Function]:
        """
        Return CasADi-wrapped functions A_d(x,u), B_d(x,u) by computing
        ZOH-discretized matrices using scipy.linalg.expm (safe, plugin-free).
        """

        # MX symbolic variables
        x = ca.MX.sym("x", self.dim_states)
        u = ca.MX.sym("u", self.dim_inputs)

        f = self.dynamics_function(x, u)

        A_c = ca.jacobian(f, x)
        B_c = ca.jacobian(f, u)

        A_func = ca.Function("A_c_func", [x, u], [A_c])
        B_func = ca.Function("B_c_func", [x, u], [B_c])

        def compute_discrete_matrices(x_val, u_val):
            A_val = np.array(A_func(x_val, u_val))
            B_val = np.array(B_func(x_val, u_val))

            aug = np.zeros((self.dim_states + self.dim_inputs, self.dim_states + self.dim_inputs))
            aug[:self.dim_states, :self.dim_states] = A_val
            aug[:self.dim_states, self.dim_states:] = B_val

            expm_matrix = scipy.linalg.expm(aug * dt)
            A_d_val = expm_matrix[:self.dim_states, :self.dim_states]
            B_d_val = expm_matrix[:self.dim_states, self.dim_states:]

            return A_d_val, B_d_val

        # CasADi external function wrapping NumPy+SciPy code
        A_d_func = ca.external("A_d_func", lambda x_, u_: compute_discrete_matrices(x_, u_)[0])
        B_d_func = ca.external("B_d_func", lambda x_, u_: compute_discrete_matrices(x_, u_)[1])

        return A_d_func, B_d_func

    
    def get_equilibrium_input(self, state: np.ndarray) -> float:

        p, v = state
        
        '''
        u = ca.SX.sym("u")
        state_sym = ca.vertcat(ca.SX.sym("p"), ca.SX.sym("v"))
        
        # Define equilibrium_condition: dv/dt = 0
        dynamics_output = self.dynamics_function(state_sym, ca.vertcat(u))
        dvdt = dynamics_output[1] # extract v_dot

        # Substitue the value into symbolic variable to get equilibrium equation
        equilibrium_condition = ca.substitute(dvdt, state_sym, ca.vertcat(p, v))

        # Solve equilibrium equation to get u_eq
        u_eq = ca.solve(equilibrium_condition, u)
        '''

        theta = self.env.theta(p)

        u_eq = 9.81 * np.sin(theta)

        return float(u_eq)
    
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

        if self.env.disturbance_lbs is not None and self.env.disturbance_lbs is not None:
            next_state += np.random.uniform(self.env.disturbance_lbs, self.env.disturbance_ubs) * dt

        return next_state



class InputDynamics:
    def __init__(self,
                 param_id: np.array = None,
                 ):
        
        # acc = 1 / (s * params[0] + params[1]) * acc_cmd
        self.params = param_id 
        
        # Define input dynamics
        if self.params is not None:
            self.input_function = self.setup_input() #u_dot = self.input_function(u, u_cmd)
        else:
            raise ValueError("Error: choose to use the identified input model but no identified parameters are given!")
    
    def setup_input(self):

        # Assume a model of the form:
        # acc = 1 / (s * params[0] + params[1]) * acc_cmd
        # acc_cmd = (s * params[0] + params[1]) * acc
        #         = params[0] * jerk + params[1] * acc 
        #         = [jerk, acc] * [params[0]; params[1]]

        u = ca.SX.sym("u")
        u_cmd = ca.SX.sym("u_cmd")

        # Expression of dynamics
        dudt = (1/self.params[0]) * u_cmd - (self.params[1]/self.params[0]) * u

        state = ca.vertcat(u)
        input = ca.vertcat(u_cmd)
        rhs = ca.vertcat(dudt)

        return ca.Function("input_function", [state, input], [rhs])
    
    def one_step_forward(self, current_input: np.ndarray, input_cmd: np.ndarray, dt: float) -> np.ndarray:

        '''use current input, input_cmd and time difference to calculate next state'''

        current_input = np.ravel(current_input)
        input_cmd = np.ravel(input_cmd)

        #print(f"current_input: {current_input}")
        #print(f"input_cmd: {input_cmd}")
        
        t_span = [0, dt]
        sim_dynamics = lambda t, state: self.input_function(state, [input_cmd]).full().flatten()

        solution = solve_ivp(
            sim_dynamics,
            t_span,
            current_input,
            method='RK45',
            t_eval=[dt]
        )
        
        new_input = np.array(solution.y)[:, -1]

        return new_input





class Env_rl_d(Env):
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
            reward = -1
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
    


class Env_rl_c(Env):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics = None,
            dt: float = 0.1
        ) -> None:

        self.env = env
        
        super().__init__(self.env.case, self.env.init_state, self.env.target_state, 
                         state_lbs=self.env.state_lbs, state_ubs=self.env.state_ubs, 
                         input_lbs=self.env.input_lbs, input_ubs=self.env.input_ubs)

        if self.state_lbs is None or self.state_ubs is None or self.input_lbs is None or self.input_ubs is None:
            raise ValueError("Constraints on states and input must been fully specified!")
        
        # self.pos_lbs, self.pos_ubs
        # self.vel_lbs, self.vel_ubs
        # self.input_lbs, self.input_ubs
        
        # State propagation
        self.dynamics = dynamics
        self.dt = dt

    def one_step_forward(self, cur_state, cur_input):
        
        # Propagate the state
        next_state = self.dynamics.one_step_forward(cur_state, cur_input, self.dt)  
        next_state_raw = next_state

        # Check whether next state is within the state space
        next_pos = max(min(next_state[0], self.pos_ubs), self.pos_lbs)
        next_vel = max(min(next_state[1], self.vel_ubs), self.vel_lbs)
        next_state = np.array([next_pos, next_vel])
        
        # Get reward
        if np.linalg.norm(next_state[0]-self.env.target_state[0])<5e-2:
            done = True
            reward = 10.0
        elif np.any(next_state_raw != next_state):
            done = True
            reward = -5.0
        else:
            done = False
            reward = np.exp( - 1.0 * np.linalg.norm(next_pos-self.env.target_state[0]))


        
        
        return done, next_state, reward




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
        result = compute_action(self, current_state, current_time)
        u = result if not isinstance(result, tuple) else result[0] # incase mpc (contains predicted trajs)

        # Check whether input is compatible with given constraints
        if input_lbs is not None:

            if not np.all(input_lbs <= u):
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
        
        if isinstance(result, tuple):
            return (u,) + result[1:]
        else:
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

# Implementation of approximate dynamic programming (ADP) controller
# Can only be used for linear system
# Implementation based on state discretization and interpolation
class ADPController(BaseController):
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
        self.num_x1 = 40
        self.num_x2 = 40
        
        # Initialize state and input space
        self.X1 = None
        self.X2 = None
        
        # Initialize policy and cost-to-go
        self.U = None
        self.J = None
    
        self.setup()

    def terminal_cost(self, x):

        return (x-self.target_state).T @ self.Qf @ (x-self.target_state)

    def stage_cost(self, x, u):

        return (x-self.target_state).T @ self.Q @ (x-self.target_state) + self.R * u**2

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
        self.X1 = np.linspace(self.env.state_lbs[0], self.env.state_ubs[0], self.num_x1)
        self.X2 = np.linspace(self.env.state_lbs[1], self.env.state_ubs[1], self.num_x2)
        
        # Initialize policy and cost-to-go function, and generate interpolation over grid mesh
        self.U = np.zeros((self.num_x1, self.num_x2))
        self.U_func = interp2d(self.X1, self.X2, self.U.T, kind='cubic')
        self.J = np.zeros((self.num_x1, self.num_x2))
        for i in range(self.num_x1): # Initialize cost matirx with terminal cost
            for j in range(self.num_x2):
                self.J[i, j] = self.terminal_cost(np.array([self.X1[i], self.X2[j]]))
        self.J_func = interp2d(self.X1, self.X2, self.J.T, kind='cubic')
        
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
            self.J_func = interp2d(self.X1, self.X2, self.J.T, kind='cubic')
            self.U_func = interp2d(self.X1, self.X2, self.U.T, kind='cubic')
        
        if self.verbose:
            print(f"Dynamic Programming policy with input constraints computed.")
            print(f"Optimal control input: {self.U}")
            print(f"Optimal cost-to-go function: {self.J}")

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_time: int) -> np.ndarray:

        if np.all(self.U == 0):
            raise ValueError("DP Policy is not computed. Call setup() first.")
        
        # Method 1: use linear interpolation to find the value of U at the current state
        miu = self.U_func(current_state[0], current_state[1]) # interpolation error will degrade the performance

        # Method 2: find the NN for state in grid, and use the related U as input
        #nearest_x1_index = np.argmin(np.abs(self.X1 - current_state[0]))
        #nearest_x2_index = np.argmin(np.abs(self.X2 - current_state[1]))
        #miu = self.U[nearest_x1_index, nearest_x2_index]
        
        #if self.verbose:
        print(f"state: {current_state}, input: {miu}")

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


# Implementation of dynamic programming (DP) controller
# Can only be used for linear system
# Implementation based on sympy
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
            verbose: bool = False,
            bangbang: bool = False,
            symbolic_weight: bool = False
        ) -> None:

        """
        DP solver for linear system:
            x_{k+1} = A x_k + B u_k
            cost = sum u_k^T R u_k + terminal cost
        """

        super().__init__(env, dynamics, freq, name, verbose)

        self.N = Horizon

        self.Q = Q
        self.R = R
        self.Qf = Qf

        # Solve input at equilibrium
        self.u_eq = self.dynamics.get_equilibrium_input(self.init_state)

        # Get linearized & discretized A and B matrices
        self.Ad, self.Bd = self.dynamics.get_linearized_AB_discrete(self.init_state, 0, self.dt)
        # siqi's case:
        #self.Ad = sp.Matrix([[1, 0], [0, 0]])
        #self.Bd = sp.Matrix([[1], [0]])
        # mountain car case:
        #self.Ad = sp.Matrix([[1, 1], [0, 1]])
        #self.Bd = sp.Matrix([[0], [1]])

        self.bangbang = bangbang # True or False
        self.symbolic_weight = symbolic_weight # True or False

        # Define symbolic variables
        self.x_sym = None
        self.u_sym = None
        self.Q_sym = None
        self.R_sym = None
        self.Qf_sym = None
        self.x_ref_sym = None

        # Store results
        self.J_sym = [None] * (self.N + 1)
        self.mu_sym = [None] * self.N
    
        self.setup()

    def setup(self) -> None:

        # Create symbolic states and inputs
        self.x_sym = [sp.Matrix(sp.symbols(f'p_{k} v_{k}')) for k in range(self.N + 1)]
        self.u_sym = [sp.Symbol(f'u_{k}') for k in range(self.N)]
        # Create symbolic weight matrices
        if self.symbolic_weight:
            q1, q2 = sp.symbols('q_p q_v')
            self.Q_sym = sp.diag(q1, q2)
            self.R_sym = sp.Symbol('r')
            self.Qf_sym = sp.diag(q1, q2)
            # Create symbolic reference state
            p_ref, v_ref = sp.symbols('p_ref v_ref')
            x_ref = [p_ref, v_ref]
            self.x_ref_sym = sp.Matrix(x_ref) 
        else:
            self.Q_sym = sp.Matrix(self.Q)
            self.R_sym = sp.Float(self.R)
            self.Qf_sym = sp.Matrix(self.Qf)
            # Create numpy reference state
            self.x_ref_sym = sp.Matrix(self.target_state) 

        # Make a copy
        J, mu = self.J_sym, self.mu_sym

        # Terminal cost: J_N = (x_N - x_ref)^T Qf (x_N - x_ref)
        err_N = self.x_sym[self.N] - self.x_ref_sym
        J[self.N] = (err_N.T * self.Qf_sym * err_N)[0, 0]

        for k in reversed(range(self.N)):

            # x_{k+1} = A x_k + B u_k
            x_next = self.Ad * self.x_sym[k] + self.Bd * self.u_sym[k]

            # Cost at step k: u_k^T R u_k + J_{k+1}(x_{k+1})
            stage_cost = self.R_sym * self.u_sym[k]**2
            J_kplus1_sub = J[k + 1].subs({self.x_sym[k + 1][i]: x_next[i] for i in range(2)})

            total_cost = stage_cost + J_kplus1_sub

            # Compute the optimal control input and cost-to-go
            if self.bangbang: # bangbang input (MIP)
                mu_k, J_k = self.solve_bangbang(total_cost, k)
            else: # continious input (NLP)
                mu_k, J_k = self.solve_continuous(total_cost, k)
            
            # Store the symbolic expressions
            mu[k] = mu_k
            J[k] = J_k
        
        # Log the symbolic expressions back
        self.J_sym = J
        self.mu_sym = mu

        if self.verbose:
            print(f"Dynamic Programming policy with input constraints computed.")
            self.print_solution()

    def solve_continuous(self, total_cost, k):

        # Derivative w.r.t u_k
        dJ_du = sp.diff(total_cost, self.u_sym[k])
        u_star = sp.solve(dJ_du, self.u_sym[k])[0]
        mu_k = sp.simplify(u_star)

        # Plug u_k* back into cost to get J_k
        cost_k_opt = total_cost.subs(self.u_sym[k], u_star)
        J_k = sp.simplify(cost_k_opt)

        return mu_k, J_k
    
    def solve_bangbang(self, total_cost, k):

        # Evaluate cost for u_k = -1 and u_k = 1
        cost_minus1 = sp.simplify(total_cost.subs(self.u_sym[k], -1))
        cost_plus1  = sp.simplify(total_cost.subs(self.u_sym[k], 1))

        # Try subtracting to simplify the condition
        delta_cost = sp.simplify(cost_plus1 - cost_minus1)

        # Store optimal control policy as piecewise
        mu_k = sp.Piecewise(
            (-1, delta_cost > 0),
            (1,  True)  # fallback
        )
        # OR mu_k = sp.simplify(mu_k)
        # OR mu_k = sp.piecewise_fold(mu_k)

        # Store cost-to-go as piecewise
        J_k = sp.Piecewise(
            (cost_minus1, delta_cost > 0),
            (cost_plus1,  True)
        )
        # OR J_k = sp.simplify(J_k)
        # OR J_k = sp.piecewise_fold(J_k)

        return mu_k, J_k

    def print_solution(self):
        for k, uk in enumerate(self.mu_sym):
            print(f"u_{k}*(x_{k}) =", uk)
        #print("\nJ_0(x_0) =")
        #sp.pprint(self.J_sym[0])

    def save_policy(self, filename='dp_policy.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({
                'mu_sym': self.mu_sym,
                'J_sym': self.J_sym,
            }, f)

    def load_policy(self, filename='dp_policy.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.mu_sym = data['mu_sym']
            self.J_sym = data['J_sym']

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_step: int) -> np.ndarray:

        if any(mu is None for mu in self.mu_sym):
            raise ValueError("DP Policy is not computed yet. Call setup() first.")
        
        if current_step >= self.N:
            raise ValueError(f"current_step = {current_step} exceeds or equals horizon N = {self.N}")
        
        mu_expr = self.mu_sym[current_step]

        # Substitute current state into the symbolic expression
        if self.symbolic_weight:
            subs_dict = {
                sp.Symbol(f'p_{current_step}'): current_state[0],
                sp.Symbol(f'v_{current_step}'): current_state[1],
                sp.Symbol('q_p'): self.Q[0, 0],
                sp.Symbol('q_v'): self.Q[1, 1],
                sp.Symbol('r'): self.R[0, 0],
                sp.Symbol('p_ref'): self.target_state[0], 
                sp.Symbol('v_ref'): self.target_state[1]
            }
        else:
            subs_dict = {
                sp.Symbol(f'p_{current_step}'): current_state[0],
                sp.Symbol(f'v_{current_step}'): current_state[1],
            }
    
        mu = float(mu_expr.subs(subs_dict).evalf())

        mu += self.u_eq  # Add equilibrium input

        if self.verbose:
            print(f"state: {current_state}, optimal input: {mu}")

        return mu
    

# Derived class for LQR Controller with finite horizon
class FiniteLQRController(BaseController):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Q_N: np.ndarray, 
            freq: float, 
            horizon: int,
            name: str = 'LQR_finite', 
            type: str = 'LQR_finite', 
            verbose: bool = False
        ) -> None:

        super().__init__(env, dynamics, freq, name, type, verbose)
        
        # Initialize as private property
        self._Q = None
        self._R = None
        self._Q_N = None

        self.N = horizon

        self.K_list = [None] * self.N  # LQR gain matrix
        
        # Call setter for the check and update the value of private property
        self.Q = Q
        self.R = R
        self.Q_N = Q_N

        self.A = None  # State transfer matrix
        self.B = None  # Input matrix

        self.x_eq = None  # Equilibrium state
        self.u_eq = None  # Equilibrium input

        self.state_lin = self.target_state
        
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
    def Q_N(self) -> np.ndarray:
        return self._Q_N

    @Q_N.setter
    def Q_N(self, value: np.ndarray) -> None:

        is_square(value)
        is_symmetric(value)
        is_positive_semi_definite(value)

        if self.verbose:
            print("Check passed, Q_N is a symmetric, positive semi-definite matrix.")

        self._Q_N = value

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
    
    def setup(self) -> None:
        
        # Set up equilibrium state
        # Note that if target state is not on the slope, self.u_eq = 0 -> will not work for the nonlinear case
        self.x_eq = self.state_lin

        # Solve input at equilibrium
        self.u_eq = self.dynamics.get_equilibrium_input(self.x_eq)

        # Linearize dynamics at equilibrium
        self.A, self.B = self.dynamics.get_linearized_AB_discrete(
            current_state=self.x_eq, current_input=self.u_eq, dt=self.dt
        )

        # Initialize terminal cost
        P = self.Q_N.copy()

        # Solve Bellman Recursion from backwardsto compute gain matrix
        for k in reversed(range(self.N)):
            K = - np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
            self.K_list[k] = K
            P = self.Q + self.A.T @ P @ self.A + self.A.T @ P @ self.B @ K

        if self.verbose:
            print(f"LQR Gain Matrix K: {self.K}")

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_time: int) -> np.ndarray:
        if any(k is None for k in self.K_list):
            raise ValueError("LQR gain matrix K is not computed. Call setup() first.")
        
        k = current_time  # assume current_time in [0, N-1]

        if k >= self.N:
            k = self.N - 1  # use terminal gain if past horizon

        # Compute state error
        det_x = current_state - self.target_state

        # Get the corresponding gain matrix for the current time step
        K_k = self.K_list[k]

        # Apply control law
        u = self.u_eq + K_k @ det_x

        return u


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
            verbose: bool = False
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

        self.x_eq = None  # Equilibrium state
        self.u_eq = None  # Equilibrium input

        self.state_lin = self.target_state
        
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

        eigvals = np.linalg.eigvals(self.A + self.B @ value)

        if np.any(np.abs(eigvals) > 1):
            raise ValueError("Warning: not all eigenvalue of A_cl inside unit circle, close-loop system is unstable!")
        
        elif self.verbose:
            print(f"Check passed, current gain K={value}, close-loop system is stable.")

        self._K = value
    
    def set_lin_point(self, state_lin: np.ndarray) -> None:

        self.state_lin = state_lin
        
        # Refresh
        self.setup()

    def setup(self) -> None:
        
        # Set up equilibrium state
        # Note that if target state is not on the slope, self.u_eq = 0 -> will not work for the nonlinear case
        self.x_eq = self.state_lin

        # Solve input at equilibrium
        self.u_eq = self.dynamics.get_equilibrium_input(self.x_eq)

        # Linearize dynamics at equilibrium
        self.A, self.B = self.dynamics.get_linearized_AB_discrete(
            current_state=self.x_eq, current_input=self.u_eq, dt=self.dt
        )

        # Solve DARE to compute gain matrix
        P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = - np.linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

        if self.verbose:
            print(f"LQR Gain Matrix K: {self.K}")

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_time) -> np.ndarray:
        if self.K is None:
            raise ValueError("LQR gain matrix K is not computed. Call setup() first.")

        # Compute state error
        det_x = current_state - self.target_state

        # Apply control law
        u = self.u_eq + self.K @ det_x

        #print(f"self.u_eq: {self.u_eq}")
        #print(f"self.K: {self.K}")
        #print(f"det_x: {det_x}")
        #print(f"self.K @ det_x: {self.K @ det_x}")

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
            max_iter: int = 100, 
            tol: float = 1e-1, 
            verbose: bool = True
        ) -> None:

        self.Qf = Qf  # Terminal cost matrix

        self.max_iter = max_iter
        self.tol = tol

        self.K_k_arr = None
        self.u_kff_arr = None
        self.total_cost_list = []  # Store total cost per iteration

        super().__init__(env, dynamics, Q, R, freq, name, type, verbose)

    def setup(self, input_traj: np.ndarray) -> None:
        """
        Perform iLQR to compute the optimal control sequence.
        """
        self.x_eq = self.target_state
        self.u_eq = self.dynamics.get_equilibrium_input(self.x_eq)

        N = len(input_traj)

        x_traj = np.zeros((self.dim_states, N+1))
        u_traj = np.copy(input_traj)
        x_traj[:, 0] = self.init_state

        self.K_k_arr = np.zeros((self.dim_states, N))
        self.u_kff_arr = np.zeros((N,))

        for n in range(self.max_iter):
            for k in range(N):
                next_state = self.dynamics.one_step_forward(current_state=x_traj[:, k], current_input=u_traj[k], dt=self.dt)
                x_traj[:, k + 1] = next_state

            x_N_det = x_traj[:, -1] - self.target_state
            x_N_det = x_N_det.reshape(-1, 1)

            s_k_bar = (x_N_det.T @ self.Qf @ x_N_det) / 2
            s_k = self.Qf @ x_N_det
            S_k = self.Qf

            for k in range(N - 1, -1, -1):
                A_lin, B_lin = self.dynamics.get_linearized_AB_discrete(current_state=x_traj[:, k], current_input=u_traj[k], dt=self.dt)

                x_k_det = x_traj[:, k] - self.target_state
                x_k_det = x_k_det.reshape(-1, 1)
                
                g_k_bar = (x_k_det.T @ self.Q @ x_k_det + self.R * u_traj[k] ** 2) * self.dt / 2
                q_k = (self.Q @ x_k_det) * self.dt
                Q_k = (self.Q) * self.dt
                r_k = (self.R * u_traj[k]) * self.dt
                R_k = (self.R) * self.dt
                P_k = np.zeros((2,)) * self.dt

                l_k = (r_k + B_lin.T @ s_k)
                G_k = (P_k + B_lin.T @ S_k @ A_lin)
                H_k = (R_k + B_lin.T @ S_k @ B_lin)

                det_u_kff = - np.linalg.inv(H_k) @ l_k
                K_k = - np.linalg.inv(H_k) @ G_k
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
                
            new_u_traj = np.zeros_like(u_traj)
            new_x_traj = np.zeros_like(x_traj)
            new_x_traj[:, 0] = self.init_state

            for k in range(N):
                new_u_traj[k] = self.u_kff_arr[k] + self.K_k_arr[:, k].T @ new_x_traj[:, k]
                next_state = self.dynamics.one_step_forward(current_state=new_x_traj[:, k], current_input=new_u_traj[k], dt=self.dt)
                new_x_traj[:, k + 1] = next_state

            # ---- Compute total cost for this iteration ----
            total_cost = 0.0
            for k in range(N):
                x_k_det = x_traj[:, k] - self.target_state
                total_cost += 0.5 * (x_k_det.T @ self.Q @ x_k_det + self.R * u_traj[k] ** 2)
            x_N_det = x_traj[:, -1] - self.target_state
            total_cost += 0.5 * (x_N_det.T @ self.Qf @ x_N_det)
            self.total_cost_list.append(total_cost.item())

            if np.max(np.abs(new_u_traj - u_traj)) < self.tol:
                print(f"Use {n} iteration until converge.")
                break
            else:
                print(f"Iteration {n}: residual error is {np.max(np.abs(new_u_traj - u_traj))}")

            u_traj = new_u_traj
            x_traj = new_x_traj

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_step: int) -> np.ndarray:
        if self.K_k_arr is None or self.u_kff_arr is None:
            raise ValueError("iLQR parameters are not computed. Call setup() first.")

        u = self.u_kff_arr[current_step] + self.K_k_arr[:, current_step].T @ current_state
        return u



# Class for linear OCP Controller
class LinearOCPController:
    def __init__(self, env, dynamics, Q, R, Qf, freq: float, N: int, name='OCP-2-norm', type='OCP', verbose=False):
        self.env = env
        self.dynamics = dynamics
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.freq = freq
        self.dt = 1.0 / freq
        self.N = N
        self.name = name
        self.type = type
        self.verbose = verbose

        self.dim_states = dynamics.dim_states
        self.dim_inputs = dynamics.dim_inputs

        self.u_seq = None

    def compute_action(self, current_state, current_time):
        if self.u_seq is None:

            start_time = time.time()

            self._solve_ocp(current_state)

            print(f"Computation time: {time.time()-start_time}")

        index = current_time
        if index < self.u_seq.shape[0]:
            return self.u_seq[index]
        else:
            return self.u_seq[-1]

    def _solve_ocp(self, current_state):
        x0 = current_state.reshape(-1, 1)
        x_ref = self.env.target_state.reshape(-1, 1)
        u_ref = np.array(self.dynamics.get_equilibrium_input(x_ref)).reshape(-1, 1)

        A_d, B_d = self.dynamics.get_linearized_AB_discrete(x0, u_ref, self.dt)

        nx = self.dim_states
        nu = self.dim_inputs
        N = self.N
        n_vars = (N + 1) * nx + N * nu

        # Cost
        H = np.zeros((n_vars, n_vars))
        f = np.zeros((n_vars, 1))

        for k in range(N):
            idx_x = slice(k * nx, (k + 1) * nx)
            idx_u = slice((N + 1) * nx + k * nu, (N + 1) * nx + (k + 1) * nu)
            H[idx_x, idx_x] = self.Q
            f[idx_x] = -self.Q @ x_ref
            H[idx_u, idx_u] = self.R
            f[idx_u] = -self.R @ u_ref

        idx_terminal = slice(N * nx, (N + 1) * nx)
        H[idx_terminal, idx_terminal] = self.Qf
        f[idx_terminal] = -self.Qf @ x_ref

        # Dynamics constraints
        Aeq = []
        beq = []

        for k in range(N):
            row = np.zeros((nx, n_vars))
            idx_xk = slice(k * nx, (k + 1) * nx)
            idx_xkp1 = slice((k + 1) * nx, (k + 2) * nx)
            idx_uk = slice((N + 1) * nx + k * nu, (N + 1) * nx + (k + 1) * nu)
            row[:, idx_xk] = A_d
            row[:, idx_uk] = B_d
            row[:, idx_xkp1] = -np.eye(nx)
            Aeq.append(row)
            beq.append(np.zeros((nx, 1)))

        row0 = np.zeros((nx, n_vars))
        row0[:, 0:nx] = np.eye(nx)
        Aeq.insert(0, row0)
        beq.insert(0, x0)

        Aeq = np.vstack(Aeq)
        beq = np.vstack(beq)

        # Inequality constraints
        G = []
        h = []

        for k in range(N + 1):
            idx_x = slice(k * nx, (k + 1) * nx)
            if self.env.state_lbs is not None:
                Gx_l = np.zeros((nx, n_vars))
                Gx_l[:, idx_x] = -np.eye(nx)
                G.append(Gx_l)
                h.append(-np.array(self.env.state_lbs).reshape(-1, 1))
            if self.env.state_ubs is not None:
                Gx_u = np.zeros((nx, n_vars))
                Gx_u[:, idx_x] = np.eye(nx)
                G.append(Gx_u)
                h.append(np.array(self.env.state_ubs).reshape(-1, 1))

        for k in range(N):
            idx_u = slice((N + 1) * nx + k * nu, (N + 1) * nx + (k + 1) * nu)
            if self.env.input_lbs is not None:
                Gu_l = np.zeros((nu, n_vars))
                Gu_l[:, idx_u] = -np.eye(nu)
                G.append(Gu_l)
                h.append(-np.array(self.env.input_lbs).reshape(-1, 1))
            if self.env.input_ubs is not None:
                Gu_u = np.zeros((nu, n_vars))
                Gu_u[:, idx_u] = np.eye(nu)
                G.append(Gu_u)
                h.append(np.array(self.env.input_ubs).reshape(-1, 1))

        G = np.vstack(G) if G else np.zeros((1, n_vars))
        h = np.vstack(h) if h else np.array([[1e10]])

        solvers.options['show_progress'] = self.verbose
        sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h), matrix(Aeq), matrix(beq))
        z_opt = np.array(sol['x']).flatten()

        u_start = (N + 1) * nx
        self.u_seq = z_opt[u_start:].reshape(N, nu) + u_ref



class LinearOCP1NormController:
    def __init__(self, env, dynamics, freq: float, N: int, name='OCP-1-norm', type='OCP', verbose=False):
        self.env = env
        self.dynamics = dynamics
        self.freq = freq
        self.dt = 1.0 / freq
        self.N = N
        self.name = name
        self.type = type
        self.verbose = verbose

        self.dim_states = dynamics.dim_states
        self.dim_inputs = dynamics.dim_inputs

        self.u_seq = None  # open-loop control sequence

    def compute_action(self, current_state, current_time):
        if self.u_seq is None:
            self._solve_ocp(current_state)

        index = current_time
        if index < self.u_seq.shape[0]:
            return self.u_seq[index]
        else:
            return self.u_seq[-1]

    def _solve_ocp(self, current_state):
        x0 = current_state.reshape(-1, 1)
        u0 = np.zeros((self.dim_inputs, 1))
        A_d, B_d = self.dynamics.get_linearized_AB_discrete(x0, u0, self.dt)

        nx = self.dim_states
        nu = self.dim_inputs
        N = self.N
        n_vars = nx * (N + 1) + nu * N + nu * N  # x, u, t

        def idx_x(k): return slice(k * nx, (k + 1) * nx)
        def idx_u(k): return slice((N + 1) * nx + k * nu, (N + 1) * nx + (k + 1) * nu)
        def idx_t(k): return slice((N + 1) * nx + N * nu + k * nu, (N + 1) * nx + N * nu + (k + 1) * nu)

        # Cost: sum of slack variables t_k
        H = np.zeros((n_vars, n_vars))
        f = np.zeros((n_vars, 1))
        for k in range(N):
            f[idx_t(k)] = 1.0

        # Equality constraints
        Aeq = []
        beq = []

        for k in range(N):
            row = np.zeros((nx, n_vars))
            row[:, idx_x(k)] = A_d
            row[:, idx_u(k)] = B_d
            row[:, idx_x(k+1)] = -np.eye(nx)
            Aeq.append(row)
            beq.append(np.zeros((nx, 1)))

        row0 = np.zeros((nx, n_vars))
        row0[:, idx_x(0)] = np.eye(nx)
        Aeq.insert(0, row0)
        beq.insert(0, x0)

        xg = self.env.target_state.reshape(-1, 1)
        rowT = np.zeros((nx, n_vars))
        rowT[:, idx_x(N)] = np.eye(nx)
        Aeq.append(rowT)
        beq.append(xg)

        Aeq = np.vstack(Aeq)
        beq = np.vstack(beq)

        # Inequality constraints
        G = []
        h = []

        for k in range(N):
            G1 = np.zeros((nu, n_vars))
            G1[:, idx_u(k)] = np.eye(nu)
            G1[:, idx_t(k)] = -np.eye(nu)
            G.append(G1)
            h.append(np.zeros((nu, 1)))

            G2 = np.zeros((nu, n_vars))
            G2[:, idx_u(k)] = -np.eye(nu)
            G2[:, idx_t(k)] = -np.eye(nu)
            G.append(G2)
            h.append(np.zeros((nu, 1)))

            if self.env.input_ubs is not None:
                G3 = np.zeros((nu, n_vars))
                G3[:, idx_u(k)] = np.eye(nu)
                G.append(G3)
                h.append(np.array(self.env.input_ubs).reshape(-1, 1))

            if self.env.input_lbs is not None:
                G4 = np.zeros((nu, n_vars))
                G4[:, idx_u(k)] = -np.eye(nu)
                G.append(G4)
                h.append(-np.array(self.env.input_lbs).reshape(-1, 1))

        G = np.vstack(G)
        h = np.vstack(h)

        # Solve QP
        solvers.options['show_progress'] = self.verbose
        sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h), matrix(Aeq), matrix(beq))
        z_opt = np.array(sol['x']).flatten()

        u_start = (N + 1) * nx
        self.u_seq = z_opt[u_start:u_start + N * nu].reshape(N, nu)

    


# Class for linear OCP Controller with inf-norm cost (open-loop version)
class LinearOCPInfNormController:
    def __init__(self, env: Env, dynamics: Dynamics, freq: float, N: int,
                 name: str = 'OCP-inf-norm', type: str = 'OCP', verbose: bool = False) -> None:
        self.env = env
        self.dynamics = dynamics
        self.N = N
        self.name = name
        self.type = type
        self.freq = freq
        self.dt = 1.0 / freq
        self.verbose = verbose

        self.dim_states = dynamics.dim_states
        self.dim_inputs = dynamics.dim_inputs

        self.u_seq = None  # store open-loop sequence

    @check_input_constraints
    def compute_action(self, current_state, current_time: int):
        if self.u_seq is None:
            self._solve_ocp(current_state)

        if current_time < len(self.u_seq):
            return self.u_seq[current_time]
        else:
            return self.u_seq[-1]  # repeat last action if time exceeds

    def _solve_ocp(self, current_state):
        x0 = current_state.reshape(-1, 1)
        u0 = np.zeros((self.dim_inputs, 1))

        A_d, B_d = self.dynamics.get_linearized_AB_discrete(x0, u0, self.dt)

        n_vars = self.dim_states * (self.N + 1) + self.dim_inputs * self.N + 1
        def idx_x(k): return slice(k * self.dim_states, (k + 1) * self.dim_states)
        def idx_u(k): return slice((self.N + 1) * self.dim_states + k * self.dim_inputs,
                                   (self.N + 1) * self.dim_states + (k + 1) * self.dim_inputs)
        idx_t = n_vars - 1

        H = np.zeros((n_vars, n_vars))
        f = np.zeros((n_vars, 1))
        f[idx_t] = 1.0

        Aeq, beq = [], []

        for k in range(self.N):
            row = np.zeros((self.dim_states, n_vars))
            row[:, idx_x(k)] = A_d
            row[:, idx_u(k)] = B_d
            row[:, idx_x(k + 1)] = -np.eye(self.dim_states)
            Aeq.append(row)
            beq.append(np.zeros((self.dim_states, 1)))

        row0 = np.zeros((self.dim_states, n_vars))
        row0[:, idx_x(0)] = np.eye(self.dim_states)
        Aeq.insert(0, row0)
        beq.insert(0, x0)

        rowT = np.zeros((self.dim_states, n_vars))
        rowT[:, idx_x(self.N)] = np.eye(self.dim_states)
        Aeq.append(rowT)
        xg = self.env.target_state.reshape(-1, 1)
        beq.append(xg)

        Aeq = np.vstack(Aeq)
        beq = np.vstack(beq)

        G, h = [], []

        for k in range(self.N):
            for i in range(self.dim_inputs):
                row1 = np.zeros((1, n_vars))
                row1[0, idx_u(k).start + i] = 1.0
                row1[0, idx_t] = -1.0
                G.append(row1)
                h.append([0.0])

                row2 = np.zeros((1, n_vars))
                row2[0, idx_u(k).start + i] = -1.0
                row2[0, idx_t] = -1.0
                G.append(row2)
                h.append([0.0])

            if self.env.input_ubs is not None:
                row = np.zeros((self.dim_inputs, n_vars))
                row[:, idx_u(k)] = np.eye(self.dim_inputs)
                G.append(row)
                h.append(np.array(self.env.input_ubs).reshape(-1, 1))

            if self.env.input_lbs is not None:
                row = np.zeros((self.dim_inputs, n_vars))
                row[:, idx_u(k)] = -np.eye(self.dim_inputs)
                G.append(row)
                h.append(-np.array(self.env.input_lbs).reshape(-1, 1))

        G = np.vstack(G)
        h = np.vstack(h)

        solvers.options['show_progress'] = self.verbose
        sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h), matrix(Aeq), matrix(beq))
        z_opt = np.array(sol['x']).flatten()

        u_start = (self.N + 1) * self.dim_states
        self.u_seq = z_opt[u_start:u_start + self.N * self.dim_inputs].reshape(self.N, self.dim_inputs)

        if self.verbose:
            print(f"[open-loop inf-norm OCP] Optimal u_seq:\n{self.u_seq}")
    


# Class for linear MPC Controller with 2-norm cost
class LinearMPCController:
    def __init__(self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Qf: np.ndarray, 
            freq: float, 
            N: int, 
            name: str = 'LMPC', 
            type: str = 'MPC', 
            verbose: bool = False
        ) -> None:

        self.env = env
        self.dynamics = dynamics

        self.Q = Q
        self.R = R
        self.Qf = Qf

        self.N = N

        self.name = name
        self.type = type

        self.freq = freq
        self.dt = 1.0 / freq
        
        self.verbose = verbose

        self.dim_states = dynamics.dim_states
        self.dim_inputs = dynamics.dim_inputs

        self.x_pred = None
        self.u_pred = None
    
    @check_input_constraints
    def compute_action(self, current_state, current_time=None):

        x0 = current_state.reshape(-1, 1)
        u0 = np.zeros((self.dim_inputs, 1))

        A_d, B_d = self.dynamics.get_linearized_AB_discrete(x0, u0, self.dt)

        n_vars = self.dim_states * (self.N + 1) + self.dim_inputs * self.N

        # Cost
        H = np.zeros((n_vars, n_vars))
        f = np.zeros((n_vars, 1))

        x_ref = self.env.target_state.reshape(-1, 1)  # assumed constant
        u_ref = np.array(self.dynamics.get_equilibrium_input(x_ref)).reshape(-1, 1)        # assumed constant

        for k in range(self.N):
            idx_x = slice(k * self.dim_states, (k + 1) * self.dim_states)
            idx_u = slice((self.N + 1) * self.dim_states + k * self.dim_inputs,
                          (self.N + 1) * self.dim_states + (k + 1) * self.dim_inputs)

            Qk = self.Q
            H[idx_x, idx_x] = Qk
            H[idx_u, idx_u] = self.R

            f[idx_x] = -Qk @ x_ref
            f[idx_u] = -self.R @ u_ref

        # Terminal cost
        idx_x_terminal = slice(self.N * self.dim_states, (self.N + 1) * self.dim_states)
        H[idx_x_terminal, idx_x_terminal] = self.Qf
        f[idx_x_terminal] = -self.Qf @ x_ref

        # Dynamics constraints
        Aeq = []
        beq = []

        for k in range(self.N):
            row = np.zeros((self.dim_states, n_vars))
            idx_xk = slice(k * self.dim_states, (k + 1) * self.dim_states)
            idx_xk_next = slice((k + 1) * self.dim_states, (k + 2) * self.dim_states)
            idx_uk = slice((self.N + 1) * self.dim_states + k * self.dim_inputs,
                           (self.N + 1) * self.dim_states + (k + 1) * self.dim_inputs)
            row[:, idx_xk_next] = -np.eye(self.dim_states)
            row[:, idx_xk] = A_d
            row[:, idx_uk] = B_d
            Aeq.append(row)
            beq.append(np.zeros((self.dim_states, 1)))

        # Add x0 = current_state as hard equality constraint
        row0 = np.zeros((self.dim_states, n_vars))
        row0[:, :self.dim_states] = np.eye(self.dim_states)
        Aeq.insert(0, row0)
        beq.insert(0, x0)

        Aeq = np.vstack(Aeq)
        beq = np.vstack(beq)

        # Inequality constraints
        G_list = []
        h_list = []

        for k in range(self.N + 1):
            # State constraints
            if self.env.state_lbs is not None:
                Gx_l = np.zeros((self.dim_states, n_vars))
                Gx_l[:, k * self.dim_states:(k + 1) * self.dim_states] = -np.eye(self.dim_states)
                G_list.append(Gx_l)
                h_list.append(-np.array(self.env.state_lbs).reshape(-1, 1))
            if self.env.state_ubs is not None:
                Gx_u = np.zeros((self.dim_states, n_vars))
                Gx_u[:, k * self.dim_states:(k + 1) * self.dim_states] = np.eye(self.dim_states)
                G_list.append(Gx_u)
                h_list.append(np.array(self.env.state_ubs).reshape(-1, 1))

        for k in range(self.N):
            # Input constraints
            if self.env.input_lbs is not None:
                Gu_l = np.zeros((self.dim_inputs, n_vars))
                Gu_l[:, (self.N + 1) * self.dim_states + k * self.dim_inputs:
                          (self.N + 1) * self.dim_states + (k + 1) * self.dim_inputs] = -np.eye(self.dim_inputs)
                G_list.append(Gu_l)
                h_list.append(-np.array(self.env.input_lbs).reshape(-1, 1))
            if self.env.input_ubs is not None:
                Gu_u = np.zeros((self.dim_inputs, n_vars))
                Gu_u[:, (self.N + 1) * self.dim_states + k * self.dim_inputs:
                          (self.N + 1) * self.dim_states + (k + 1) * self.dim_inputs] = np.eye(self.dim_inputs)
                G_list.append(Gu_u)
                h_list.append(np.array(self.env.input_ubs).reshape(-1, 1))

        # Check if there are any constraints
        if len(G_list) > 0:
            G = np.vstack(G_list)
            h = np.vstack(h_list)
        else:
            G = np.zeros((1, n_vars))
            h = np.array([[1e10]])

        # Solve the QP
        solvers.options['show_progress'] = False
        sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h), matrix(Aeq), matrix(beq))
        z_opt = np.array(sol['x']).flatten()

        x_opt = z_opt[: (self.N + 1) * self.dim_states].reshape(self.N + 1, self.dim_states).T
        u_opt = z_opt[(self.N + 1) * self.dim_states:].reshape(self.N, self.dim_inputs).T

        if self.verbose:
            print(f"Optimal control action: {u_opt[:, 0]}")
            print(f"Predicted x: {x_opt}")
            print(f"Predicted u: {u_opt}")

        self.x_pred = x_opt
        self.u_pred = u_opt

        return u_opt[:, 0].flatten()+u_ref, x_opt.T, u_opt



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
        
        ## Model
        # Set up Acados model
        model = AcadosModel()
        model.name = self.name

        # Define model: x_dot = f(x, u)
        model.x = self.dynamics.states
        model.u = self.dynamics.inputs
        model.f_expl_expr = ca.vertcat(self.dynamics.dynamics_function(self.dynamics.states, self.dynamics.inputs))
        model.f_impl_expr = None # no needed, we already have the explicit model


        ## Optimal control problem
        # Set up Acados OCP
        ocp = AcadosOcp()
        ocp.model = model # link to the model (class: AcadosModel)
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt  # total prediction time
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" # Partially condensing interior-point method
        ocp.solver_options.integrator_type = "ERK" # explicit Runge-Kutta
        ocp.solver_options.nlp_solver_type = "SQP" # sequential quadratic programming

        # Set up other hyperparameters in SQP solving
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.nlp_solver_tol_stat = 1E-6
        ocp.solver_options.nlp_solver_tol_eq = 1E-6
        ocp.solver_options.nlp_solver_tol_ineq = 1E-6
        ocp.solver_options.nlp_solver_tol_comp = 1E-6
        
        # For debugging
        #ocp.solver_options.print_level = 2

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

        # State constraints
        ocp.constraints.idxbx = np.arange(self.dim_states)
        ocp.constraints.idxbx_e = np.arange(self.dim_states)
        if self.env.state_lbs is None and self.env.state_ubs is None:
            ocp.constraints.lbx_0 = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_0 = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx_e = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_e = np.full(self.dim_states, 1e6)
        elif self.env.state_lbs is not None and self.env.state_ubs is None:
            ocp.constraints.lbx_0 = np.array(self.env.state_lbs)
            ocp.constraints.ubx_0 = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx = np.array(self.env.state_lbs)
            ocp.constraints.ubx = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx_e = np.array(self.env.state_lbs)
            ocp.constraints.ubx_e = np.full(self.dim_states, 1e6)
        elif self.env.state_lbs is None and self.env.state_ubs is not None:
            ocp.constraints.lbx_0 = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_0 = np.array(self.env.state_ubs)
            ocp.constraints.lbx = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx = np.array(self.env.state_ubs)
            ocp.constraints.lbx_e = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        else:
            ocp.constraints.lbx_0 = np.array(self.env.state_lbs)
            ocp.constraints.ubx_0 = np.array(self.env.state_ubs)
            ocp.constraints.lbx = np.array(self.env.state_lbs)
            ocp.constraints.ubx = np.array(self.env.state_ubs)
            ocp.constraints.lbx_e = np.array(self.env.state_lbs)
            ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        
        # Input constraints
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


        ## Ocp Solver
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
        - current_time: The current time (not used in this time-invariant case).

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
        #if status != 0:
        #    raise ValueError(f"Acados solver failed with status {status}")

        # Extract the first control action
        u_optimal = self.solver.get(0, "u")

        # Extract the predictions
        x_pred = np.zeros((self.N + 1, self.dim_states))
        u_pred = np.zeros((self.N, self.dim_inputs))
        for i in range(self.N + 1):
            x_pred[i, :] = self.solver.get(i, "x")
            if i < self.N:
                u_pred[i, :] = self.solver.get(i, "u")

        if self.verbose:
            print(f"Optimal control action: {u_optimal}")
            print(f"x_pred: {x_pred}")
            print(f"u_pred: {u_pred}")

        return u_optimal, x_pred, u_pred



class TrackingMPCController(LQRController):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Qf: np.ndarray, 
            freq: float, 
            N: int, 
            traj_ref: np.ndarray = None,
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
        self.traj_ref = traj_ref

        super().__init__(env, dynamics, Q, R, freq, name, type, verbose)

    def setup(self) -> None:
        """
        Define the MPC optimization problem using Acados.
        """
        
        ## Model
        # Set up Acados model
        model = AcadosModel()
        model.name = self.name

        # Define model: x_dot = f(x, u)
        model.x = self.dynamics.states
        model.u = self.dynamics.inputs
        model.f_expl_expr = ca.vertcat(self.dynamics.dynamics_function(self.dynamics.states, self.dynamics.inputs))
        model.f_impl_expr = None # no needed, we already have the explicit model


        ## Optimal control problem
        # Set up Acados OCP
        ocp = AcadosOcp()
        ocp.model = model # link to the model (class: AcadosModel)
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt  # total prediction time
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" # Partially condensing interior-point method
        ocp.solver_options.integrator_type = "ERK" # explicit Runge-Kutta
        ocp.solver_options.nlp_solver_type = "SQP" # sequential quadratic programming

        # Set up other hyperparameters in SQP solving
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.nlp_solver_tol_stat = 1E-6
        ocp.solver_options.nlp_solver_tol_eq = 1E-6
        ocp.solver_options.nlp_solver_tol_ineq = 1E-6
        ocp.solver_options.nlp_solver_tol_comp = 1E-6
        
        # For debugging
        #ocp.solver_options.print_level = 2

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

        # State constraints
        ocp.constraints.idxbx = np.arange(self.dim_states)
        ocp.constraints.idxbx_e = np.arange(self.dim_states)
        if self.env.state_lbs is None and self.env.state_ubs is None:
            ocp.constraints.lbx_0 = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_0 = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx_e = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_e = np.full(self.dim_states, 1e6)
        elif self.env.state_lbs is not None and self.env.state_ubs is None:
            ocp.constraints.lbx_0 = np.array(self.env.state_lbs)
            ocp.constraints.ubx_0 = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx = np.array(self.env.state_lbs)
            ocp.constraints.ubx = np.full(self.dim_states, 1e6)
            ocp.constraints.lbx_e = np.array(self.env.state_lbs)
            ocp.constraints.ubx_e = np.full(self.dim_states, 1e6)
        elif self.env.state_lbs is None and self.env.state_ubs is not None:
            ocp.constraints.lbx_0 = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_0 = np.array(self.env.state_ubs)
            ocp.constraints.lbx = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx = np.array(self.env.state_ubs)
            ocp.constraints.lbx_e = np.full(self.dim_states, -1e6)
            ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        else:
            ocp.constraints.lbx_0 = np.array(self.env.state_lbs)
            ocp.constraints.ubx_0 = np.array(self.env.state_ubs)
            ocp.constraints.lbx = np.array(self.env.state_lbs)
            ocp.constraints.ubx = np.array(self.env.state_ubs)
            ocp.constraints.lbx_e = np.array(self.env.state_lbs)
            ocp.constraints.ubx_e = np.array(self.env.state_ubs)
        
        # Input constraints
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


        ## Ocp Solver
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
        - current_time: The current time (not used in this time-invariant case).

        Returns:
        - Optimal control action.
        """

        # Update initial state in the solver
        self.solver.set(0, "lbx", current_state)
        self.solver.set(0, "ubx", current_state)

        # Update reference trajectory for all prediction steps
        input_ref = np.zeros(self.dim_inputs)
        ref_length = self.traj_ref.shape[0]

        for i in range(self.N):
            index = min(current_time + i, ref_length - 1)
            state_ref = self.traj_ref[index, :self.dim_states]
            self.solver.set(i, "yref", np.concatenate((state_ref, input_ref)))
        index = min(current_time + self.N, ref_length - 1)
        terminal_state_ref = self.traj_ref[index, :self.dim_states]
        self.solver.set(self.N, "yref", terminal_state_ref)

        # Solve the MPC problem
        status = self.solver.solve()
        #if status != 0:
        #    raise ValueError(f"Acados solver failed with status {status}")

        # Extract the first control action
        u_optimal = self.solver.get(0, "u")

        # Extract the predictions
        x_pred = np.zeros((self.N + 1, self.dim_states))
        u_pred = np.zeros((self.N, self.dim_inputs))
        for i in range(self.N + 1):
            x_pred[i, :] = self.solver.get(i, "x")
            if i < self.N:
                u_pred[i, :] = self.solver.get(i, "u")

        if self.verbose:
            print(f"Optimal control action: {u_optimal}")
            print(f"x_pred: {x_pred}")
            print(f"u_pred: {u_pred}")

        return u_optimal, x_pred, u_pred
    


# Derived class for Tube-based Robust MPC Controller
class LinearRMPCController(LQRController):
    def __init__(
            self, 
            env: Env, 
            dynamics: Dynamics, 
            Q: np.ndarray, 
            R: np.ndarray, 
            Qf: np.ndarray, 
            freq: float, 
            N: int, 
            K_feedback: Optional[np.ndarray] = None,  # Feedback gain for tube
            disturbance_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (lbz, ubz) of D
            max_iter: int = 10,  # Max iterations for invariant set computation
            name: str = 'RMPC', 
            type: str = 'RMPC', 
            verbose: bool = True
        ) -> None:

        self.Qf = Qf

        self.N = N  # Prediction horizon

        self.ocp = None
        self.solver = None

        super().__init__(env, dynamics, Q, R, freq, name, type, verbose)

        x0 = np.zeros(self.dynamics.dim_states)
        u0 = np.zeros(self.dynamics.dim_inputs)
        self.A, self.B = self.dynamics.get_linearized_AB_discrete(x0, u0, self.dt)

        # Automatically solve DARE if no K provided
        if K_feedback is None:
            from scipy.linalg import solve_discrete_are
            P = solve_discrete_are(self.A, self.B, Q, R)
            self.K_feedback = -np.linalg.inv(R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        else:
            self.K_feedback = K_feedback

        # Compute Omega_tube from disturbance box D 
        if disturbance_bounds is not None:

            disturbance_lbs = disturbance_bounds[0]
            disturbance_ubs = disturbance_bounds[1]

        elif self.env.disturbance_lbs is not None and self.env.disturbance_ubs is not None:

            disturbance_lbs = self.env.disturbance_lbs
            disturbance_ubs = self.env.disturbance_ubs
            
        else:
            raise ValueError("No bounds of additive disturbances provided, can not initialize RMPC")
        
        disturbance_lbs /= freq
        disturbance_ubs /= freq
        
        self.Omega_tube = self.compute_invariant_tube(self.A + self.B @ self.K_feedback, disturbance_lbs, disturbance_ubs, max_iter=max_iter)

        # Create polytope for tighten state constraints
        dim = len(self.env.state_lbs)
        H_box = np.vstack([np.eye(dim), -np.eye(dim)])
        h_box = np.hstack([self.env.state_ubs, -self.env.state_lbs])
        self.X = pt.Polytope(H_box, h_box)
        self.X_tighten = self.X - self.Omega_tube

        # Create polytope for tighten input constraints
        u_lbs_tighten = self.env.input_lbs + np.max(self.affine_map(self.Omega_tube, self.K_feedback).V, axis=0)
        u_ubs_tighten = self.env.input_ubs + np.min(self.affine_map(self.Omega_tube, self.K_feedback).V, axis=0)
        self.U_tighten = np.array([u_lbs_tighten, u_ubs_tighten])

        self.tube_bounds_x = self.estimate_bounds_from_polytope(self.Omega_tube)
        self.tube_bounds_u = np.abs(self.K_feedback @ self.tube_bounds_x)

        if self.verbose:
            print(f"Tighten state set X-Ω: {self.X_tighten.V}")
            print(f"Tube size x: {self.tube_bounds_x}")
            print(f"Tube size u: {self.tube_bounds_u}")

        self.setup()
    
    def affine_map(self, poly: pt.Polytope, A: np.ndarray) -> pt.Polytope:
        """Compute the affine image of a polytope under x ↦ A x"""

        assert poly.V is not None, "No poly.V in Polytope! Can not apply affine map."

        V = poly.V  # vertices
        V_new = (A @ V.T).T

        return pt.Polytope(V_new)
        
    def compute_invariant_tube(self, A_cl, lbz, ubz, tol=1e-4, max_iter=10) -> pt.Polytope:
        """
        Compute the robust positive invariant set Omega_tube using Minkowski recursion.

        This implementation uses the `polytope` library's Minkowski sum and affine map.
        """

        # Step 1: Define initial disturbance set D (box)
        dim = len(lbz)

        # H-representation: H x ≤ h
        H_box = np.vstack([np.eye(dim), -np.eye(dim)])
        h_box = np.hstack([ubz, -lbz])

        # Create polytope with both H-rep and V-rep
        D = pt.Polytope(H_box, h_box)

        # Step 2: Initialize Omega := D
        Omega = D

        for i in range(max_iter):
            # Step 3: Apply affine map A_cl to Omega: A_cl * Omega
            A_Omega = self.affine_map(Omega, A_cl)

            # Step 4: Minkowski sum: Omega_next = A_Omega ⊕ D
            Omega_next = A_Omega + D

            # Step 5: Check convergence via bounding box approximation
            bounds_old = self.estimate_bounds_from_polytope(Omega)
            bounds_new = self.estimate_bounds_from_polytope(Omega_next)

            if np.allclose(bounds_old, bounds_new, atol=tol):
                return Omega_next  # Return as vertices

            Omega = Omega_next

        return Omega  # Max iteration reached, return current estimate

    def estimate_bounds_from_polytope(self, poly: pt.Polytope):
        """Estimate box bounds from polytope vertices (axis-aligned)."""
        vertices = poly.V
        return np.max(np.abs(vertices), axis=0)

    def setup(self) -> None:

        ## Model
        # Set up Acados model
        model = AcadosModel()
        model.name = self.name

        # Define model: x_dot = f(x, u)
        model.x = self.dynamics.states
        model.u = self.dynamics.inputs
        model.f_expl_expr = ca.vertcat(self.dynamics.dynamics_function(self.dynamics.states, self.dynamics.inputs))
        model.f_impl_expr = None # no needed, we already have the explicit model

        ## Optimal control problem
        # Set up Acados OCP
        ocp = AcadosOcp()
        ocp.model = model # link to the model (class: AcadosModel)
        ocp.solver_options.N_horizon = self.N
        ocp.solver_options.tf = self.N * self.dt  # total prediction time
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" # Partially condensing interior-point method
        ocp.solver_options.integrator_type = "ERK" # explicit Runge-Kutta
        ocp.solver_options.nlp_solver_type = "SQP" # sequential quadratic programming

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
        ocp.cost.yref = np.concatenate((self.target_state, np.zeros(self.dim_inputs)))
        ocp.cost.yref_e = self.target_state

        # Input constraints
        ocp.constraints.idxbu = np.arange(self.dim_inputs)

        if self.env.input_lbs is None:
            ocp.constraints.lbu = np.full(self.dim_inputs, -1e6)
        else:
            ocp.constraints.lbu = self.U_tighten[0]

        if self.env.input_ubs is None:
            ocp.constraints.ubu = np.full(self.dim_inputs, 1e6)
        else:
            ocp.constraints.ubu = self.U_tighten[1]

        # Expand initial state constraints (not here, do online)
        # Add Omega constraints on initial state x0: A x0 <= b
        ocp.dims.nh_0 = self.Omega_tube.A.shape[0]
        ocp.model.con_h_expr_0 = ca.mtimes(self.Omega_tube.A, ocp.model.x)
        ocp.constraints.lh_0 = -1e6 * np.ones(self.Omega_tube.A.shape[0])
        ocp.constraints.uh_0 = 1e6 * np.ones(self.Omega_tube.A.shape[0])  # placeholder

        # Expand tighten state constraints 
        ocp.dims.nh = self.X_tighten.A.shape[0]
        ocp.dims.nh_e = self.X_tighten.A.shape[0]
        ocp.model.con_h_expr = ca.mtimes(self.X_tighten.A, ocp.model.x)
        ocp.model.con_h_expr_e = ca.mtimes(self.X_tighten.A, ocp.model.x)
        ocp.constraints.lh = -1e6 * np.ones(self.X_tighten.A.shape[0])
        ocp.constraints.lh_e = -1e6 * np.ones(self.X_tighten.A.shape[0])
        ocp.constraints.uh = self.X_tighten.b.flatten()
        ocp.constraints.uh_e = self.X_tighten.b.flatten()

        # Recreate solver with tightened constraints
        self.ocp = ocp
        self.solver = AcadosOcpSolver(self.ocp, json_file=f"{self.name}.json", generate=True)
        
        if self.verbose:
            print("Tube-based MPC setup with constraint tightening completed.")

    @check_input_constraints
    def compute_action(self, current_state: np.ndarray, current_time) -> np.ndarray:

        # Set upper limit of convex set equality constraint on target step to be 0
        lh_dynamic = self.Omega_tube.A @ current_state + self.Omega_tube.b.flatten()
        self.solver.constraints_set(0, "uh", lh_dynamic)

        status = self.solver.solve()

        x_nominal = self.solver.get(0, "x")
        u_nominal = self.solver.get(0, "u")

        # Apply tube feedback control
        u_real = u_nominal + self.K_feedback @ (current_state - x_nominal)

        if self.verbose:
            print("Current state:", current_state)
            print("Nominal state:", x_nominal)
            print("Nominal input:", u_nominal)
            print("Tube-corrected input:", u_real)

        # Also return nominal predictions
        x_pred = np.zeros((self.N + 1, self.dim_states))
        u_pred = np.zeros((self.N, self.dim_inputs))
        for i in range(self.N + 1):
            x_pred[i, :] = self.solver.get(i, "x")
            if i < self.N:
                u_pred[i, :] = self.solver.get(i, "u")

        return u_real, x_pred, u_pred, u_nominal
    
    def plot_robust_invariant_set(self):
        """
        Plot the 2D invariant set (Ω) and its axis-aligned bounding box.
        Assumes 2D state space.
        """
        Omega_vertices = self.Omega_tube.V
        assert Omega_vertices.shape[1] == 2, "Only 2D invariant sets are supported."

        # Convex hull of the Omega polytope
        hull = ConvexHull(Omega_vertices)
        hull_pts = Omega_vertices[hull.vertices]

        # Compute axis-aligned bounding box
        min_bounds = np.min(Omega_vertices, axis=0)
        max_bounds = np.max(Omega_vertices, axis=0)
        bounding_box = np.array([
            [min_bounds[0], min_bounds[1]],
            [min_bounds[0], max_bounds[1]],
            [max_bounds[0], max_bounds[1]],
            [max_bounds[0], min_bounds[1]],
            [min_bounds[0], min_bounds[1]]
        ])

        # Plot
        plt.figure(figsize=(6, 6))
        plt.fill(hull_pts[:, 0], hull_pts[:, 1], color='skyblue', alpha=0.5, label='Ω (Invariant Set)')
        plt.plot(hull_pts[:, 0], hull_pts[:, 1], 'b-', linewidth=2)
        plt.plot(bounding_box[:, 0], bounding_box[:, 1], 'r--', linewidth=2, label='Bounding Box')

        plt.title("Invariant Set Ω and Bounding Box")
        plt.xlabel("State x₁")
        plt.ylabel("State x₂")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_tighten_state_set(self):
       
        X_tighten_vertices = self.X_tighten.V

        assert X_tighten_vertices.shape[1] == 2, "Only 2D tighten sets are supported."

        # Convex hull of the Omega polytope
        hull = ConvexHull(X_tighten_vertices)
        hull_pts = X_tighten_vertices[hull.vertices]

        # Plot the original state constraints
        X = np.array([
            [self.env.state_lbs[0], self.env.state_lbs[1]],
            [self.env.state_lbs[0], self.env.state_ubs[1]],
            [self.env.state_ubs[0], self.env.state_ubs[1]],
            [self.env.state_ubs[0], self.env.state_lbs[1]],
            [self.env.state_lbs[0], self.env.state_lbs[1]]
        ])

        # Plot
        plt.figure(figsize=(6, 6))
        plt.fill(hull_pts[:, 0], hull_pts[:, 1], color='skyblue', alpha=0.5, label='X-Ω')
        plt.plot(hull_pts[:, 0], hull_pts[:, 1], 'b-', linewidth=2)
        plt.plot(X[:, 0], X[:, 1], 'r--', linewidth=2, label='X')

        plt.title("Tighten state Set X-Ω and Original set X")
        plt.xlabel("State x₁")
        plt.ylabel("State x₂")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

    


# Derived class for RL Controller, solved by General Policy Iteration
class GPIController(BaseController):
    def __init__(self, 
                 mdp: Env_rl_d, 
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
                 mdp: Env_rl_d, 
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
                terminate_condition_1 = False #next_state[0]==self.mdp.pos_partitions[-1]
                terminate_condition_2 = False #next_state[0]==self.mdp.pos_partitions[0]
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
                 mdp: Env_rl_d, 
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
        dynamics: Dynamics = None,
        controller: BaseController = None,
        env: Env = None,
        dt: float = None,
        t_terminal: float = None,
        type: str = 'ideal', # 'ideal' OR 'identified'
        param_id: np.array = None,
        verbose: bool = False
    ) -> None:
        
        self.dynamics = dynamics
        self.controller = controller
        self.env = env

        self.type = type  # 'ideal' or 'identified'

        self.verbose = verbose

        if self.type == 'ideal':
            self.input_dynamics = None
        elif self.type == 'identified' and param_id is not None:
            self.input_dynamics = InputDynamics(param_id)
        elif self.type == 'identified' and param_id == None:
            raise ValueError("Error: choose to use the identified input model but no identified parameters are given!")
            

        # Initialize attributes only if 'env' is provided
        if env is not None:
            self.init_state = env.init_state
        else:
            self.init_state = None

        # Initialize timeline attributes only if 'dt' and 't_terminal' are provided
        if dt is not None and t_terminal is not None:
            self.t_0 = 0
            self.t_terminal = t_terminal
            self.dt = dt
            self.t_eval = np.linspace(
                self.t_0, self.t_terminal, int((self.t_terminal - self.t_0) / self.dt) + 1
            )
        else:
            self.t_0 = None
            self.t_terminal = None
            self.dt = None
            self.t_eval = None

        # Initialize recording lists
        self.state_traj = []
        self.input_traj = []
        self.nominal_input_traj = []
        self.state_pred_traj = []
        self.input_pred_traj = []
        self.cost2go_arr = None
        self.counter = 0

        # Set controller character if controller is provided
        self.controller_name = controller.name if controller is not None else None
        self.controller_type = controller.type if controller is not None else None

    def reset_counter(self) -> None:
        self.counter = 0
    
    def run_simulation(self) -> None:
        
        # Initialize state vector
        current_state = self.init_state
        self.state_traj.append(current_state)

        if self.type == 'identified':
            current_input_old = 0

        for current_time in self.t_eval[:-1]:

            # Get current state, and call controller to calculate input
            if self.controller.type == 'MPC':
                input_cmd, state_pred, input_pred = self.controller.compute_action(current_state, self.counter)
                # Log the predictions
                self.state_pred_traj.append(state_pred)
                self.input_pred_traj.append(input_pred)
            elif self.controller.type == 'RMPC':
                input_cmd, state_pred, input_pred, nominal_input = self.controller.compute_action(current_state, self.counter)
                # Log the predictions
                self.state_pred_traj.append(state_pred)
                self.input_pred_traj.append(input_pred)
            else:
                input_cmd = self.controller.compute_action(current_state, self.counter)

            # Do one-step simulation
            if self.type == 'ideal':
                current_input = input_cmd
            elif self.type == 'identified' and self.input_dynamics is not None:
                current_input = self.input_dynamics.one_step_forward(current_input_old, input_cmd, self.dt)
                current_input_old = current_input
            else:
                raise NotImplementedError("Simulation type not implemented or input dynamics not provided!")
                
            current_state = self.dynamics.one_step_forward(current_state, current_input, self.dt)
            #print(f"sim_state:{current_state}")

            # Log the results
            self.state_traj.append(current_state)
            self.input_traj.append(np.array(current_input).flatten())
            if self.controller.type == 'RMPC':
                self.nominal_input_traj.append(np.array(nominal_input).flatten())
            # TODO: also log input_cmd curve for identified case

            # Update timer
            self.counter += 1
        
        print("Simulation finished, will start plotting")
        self.reset_counter()
    
    def save(self, filename='dp_sim.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({
                'controller_name': self.controller_name,
                'controller_type': self.controller_type,
                'env': self.env,
                't_eval': self.t_eval,
                'state_traj': self.state_traj,
                'input_traj': self.input_traj,
            }, f)

    def load(self, filename='dp_sim.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

            self.controller_name = data['controller_name']
            self.controller_type = data['controller_type']

            self.env = data['env']
        
            self.t_eval = data['t_eval']
            self.t_0 = self.t_eval[0]
            self.t_terminal = self.t_eval[-1]
            self.dt = self.t_eval[1] - self.t_eval[0]
            
            self.state_traj = data['state_traj']
            self.input_traj = data['input_traj']
            self.init_state = data['state_traj'][0]

    def get_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Get state and input trajectories, return in ndarray form '''

        # Transform state and input traj from list to ndarray
        state_traj = np.array(self.state_traj)
        input_traj = np.array(self.input_traj)

        return state_traj, input_traj

    def compute_cost2go(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        Qf: np.ndarray,
        target_state: np.ndarray
    ) -> float:
        
        """can only be used for quadratic cost function"""

        # Transform state and input traj from list to ndarray
        state_traj = np.array(self.state_traj)
        input_traj = np.array(self.input_traj)

        total_cost_arr = np.zeros(len(input_traj))
        
        # Terminal cost
        x_final = state_traj[-1]
        x_err_final = x_final - target_state
        cost_terminal = 0.5 * (x_err_final.T @ Qf @ x_err_final)
        total_cost_arr[-1] = cost_terminal

        # Stage cost, backpropagation
        for k in range(len(input_traj)-2, -1, -1):
            x_err = state_traj[k] - target_state
            u = input_traj[k]

            u_cost = R * (u**2)

            cost_stage = 0.5 * (x_err.T @ Q @ x_err + u_cost)
            total_cost_arr[k] = total_cost_arr[k+1] +  cost_stage
        
        # Update the cost2go_arr
        self.cost2go_arr = total_cost_arr

        return total_cost_arr




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

        self.delta_index_pred_display = 2

        # Define setting of plots and animation
        self.color = "blue"
        self.color_list = ['red', 'green', 'yellow', 'orange']

        self.shadow_space_wide = 0.2

        self.car_length= 0.2
        self.figsize = (8, 4)
        self.refresh_rate = 30

    def display_plots(self, title = None) -> None:

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))

        # Plot predicted positions
        if self.simulator.controller_type in ['MPC', 'RMPC'] and len(self.simulator.state_pred_traj)>0:
            for i in range(len(self.simulator.state_pred_traj)):
                if i%self.delta_index_pred_display == 0:
                    state_pred_traj_curr = self.simulator.state_pred_traj[i]
                    t_eval = np.linspace(self.t_eval[i], self.t_eval[i] + self.simulator.dt * (len(state_pred_traj_curr) - 1), len(state_pred_traj_curr))
                    ax1.plot(t_eval, state_pred_traj_curr[:, 0], label="Predicted Position", linestyle="--", color="orange")
        
        # Plot reference if have
        if hasattr(self.simulator.controller, 'traj_ref'):
            ax1.plot(self.t_eval, self.simulator.controller.traj_ref[:, 0], color='gray', marker='.', linestyle='-', label='Reference Position')

        # Plot p over time t
        ax1.plot(self.t_eval, self.position, label="Position p(t)", color="blue")

        # Plot shadowed zone for state bounds
        if self.env.state_lbs is not None:
            ymin, _ = ax1.get_ylim()
            lower_edge = min(self.env.state_lbs[0] - self.shadow_space_wide, ymin)
            upper_edge = self.env.state_lbs[0]
            ax1.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'pos_lower_bound')
        if self.env.state_ubs is not None:
            _, ymax = ax1.get_ylim()
            lower_edge = self.env.state_ubs[0]
            upper_edge = max(self.env.state_ubs[0] + self.shadow_space_wide, ymax)
            ax1.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'pos_upper_bound')

        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())

        # Plot predicted velocities
        if self.simulator.controller_type in ['MPC', 'RMPC'] and len(self.simulator.state_pred_traj)>0:
            for i in range(len(self.simulator.state_pred_traj)):
                if i%self.delta_index_pred_display == 0:
                    state_pred_traj_curr = self.simulator.state_pred_traj[i]
                    t_eval = np.linspace(self.t_eval[i], self.t_eval[i] + self.simulator.dt * (len(state_pred_traj_curr) - 1), len(state_pred_traj_curr))
                    ax2.plot(t_eval, state_pred_traj_curr[:, 1], label="Predicted Velocity", linestyle="--", color="orange")
        
        # Plot reference if have
        if hasattr(self.simulator.controller, 'traj_ref'):
            ax2.plot(self.t_eval, self.simulator.controller.traj_ref[:, 1], color='gray', marker='.', linestyle='-', label='Reference Velocity')

        # Plot v over time t
        ax2.plot(self.t_eval, self.velocity, label="Velocity v(t)", color="green")

        # Plot shadowed zone for state bounds
        if self.env.state_lbs is not None:
            ymin, _ = ax2.get_ylim()
            lower_edge = min(self.env.state_lbs[1] - self.shadow_space_wide, ymin)
            upper_edge = self.env.state_lbs[1]
            ax2.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'vel_lower_bound')
        if self.env.state_ubs is not None:
            _, ymax = ax2.get_ylim()
            lower_edge = self.env.state_ubs[1]
            upper_edge = max(self.env.state_ubs[1] + self.shadow_space_wide, ymax)
            ax2.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'vel_upper_bound')

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys())

        '''
        # Not updated yet, still continuous signal instead of ZOH signal
        # Plot predicted accelerations
        if self.simulator.controller_type in ['MPC', 'RMPC'] and len(self.simulator.state_pred_traj)>0:
            for i in range(len(self.simulator.state_pred_traj)):
                if i%self.delta_index_pred_display == 0:
                    input_pred_traj_curr = self.simulator.input_pred_traj[i]
                    t_eval = np.linspace(self.t_eval[i], self.t_eval[i] + self.simulator.dt * (len(input_pred_traj_curr) - 1), len(input_pred_traj_curr))
                    ax3.plot(t_eval, input_pred_traj_curr, label="Predicted Input", linestyle="--", color="orange")
        '''

        # Plot a over time t
        #ax3.plot(self.t_eval[:-1], self.acceleration, label="Input u(t)", color="red")
        ax3.step(self.t_eval, np.append(self.acceleration, self.acceleration[-1]), where='post', label="Input u(t)", color='red')
        #ax3.plot(self.t_eval[:-1], self.acceleration, 'o', color='red')
        if self.simulator.controller_type == 'RMPC':
            nominal_acceleration = self.simulator.nominal_input_traj
            ax3.step(self.t_eval, np.append(nominal_acceleration, nominal_acceleration[-1]), where='post', label="Nominal Input u(t)", color='purple')

        # Plot shadowed zone for input bounds
        if self.env.input_lbs is not None:
            ymin, _ = ax3.get_ylim()
            lower_edge = min(self.env.input_lbs-self.shadow_space_wide, ymin)
            upper_edge = self.env.input_lbs
            ax3.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'input_lower_bound')
        if self.env.input_ubs is not None:
            _, ymax = ax3.get_ylim()
            lower_edge = self.env.input_ubs
            upper_edge = max(self.env.input_ubs+self.shadow_space_wide, ymax)
            ax3.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'input_upper_bound')
            
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Input (m/s^2)")

        handles, labels = ax3.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax3.legend(by_label.values(), by_label.keys())
        
        if title is not None:
            fig.suptitle(title, fontsize=20, fontweight='bold', y=1.05)
            
        plt.tight_layout()
        plt.show()

    def display_contrast_plots(self, *simulators: Simulator, title=None) -> None:

        color_index = 0

        if not simulators:
            raise ValueError("No simulator references provided.")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 4))

        # Plot current object's trajectories
        ax1.plot(self.t_eval, self.position, label=f"{self.simulator.controller_name}", color=self.color)

        # Plot reference if have
        if hasattr(self.simulator.controller, 'traj_ref'):
            ax1.plot(self.t_eval, self.simulator.controller.traj_ref[:, 0], color='gray', marker='.', linestyle='-', label='Reference Trajectory')

        ax2.plot(self.t_eval, self.velocity, label=f"{self.simulator.controller_name}", color=self.color)

        # Plot reference if have
        if hasattr(self.simulator.controller, 'traj_ref'):
            ax2.plot(self.t_eval, self.simulator.controller.traj_ref[:, 1], color='gray', marker='.', linestyle='-', label='Reference Trajectory')

        #ax3.plot(self.t_eval[:-1], self.acceleration, label=f"{self.simulator.controller_name}", color=self.color)
        ax3.step(self.t_eval, np.append(self.acceleration, self.acceleration[-1]), where='post', label=f"{self.simulator.controller_name}", color=self.color)

        # Plot the reference and evaluated trajectories for each simulator
        for simulator_ref in simulators:
            if not simulator_ref.state_traj:
                raise ValueError(f"Failed to get trajectory from simulator {simulator_ref.controller_name}. State trajectory list is void; please run 'run_simulation' first.")

            # Get reference trajectories from simulator_ref
            state_traj_ref, input_traj_ref = simulator_ref.get_trajectories()

            # Extract reference position, velocity, and acceleration
            position_ref = state_traj_ref[:, 0]
            velocity_ref = state_traj_ref[:, 1]
            acceleration_ref = input_traj_ref

            # Plot position over time
            ax1.plot(self.t_eval, position_ref, linestyle="--", label=f"{simulator_ref.controller_name}", color=self.color_list[color_index])

            # Plot velocity over time
            ax2.plot(self.t_eval, velocity_ref, linestyle="--", label=f"{simulator_ref.controller_name}", color=self.color_list[color_index])

            # Plot acceleration over time
            #ax3.plot(self.t_eval[:-1], acceleration_ref, linestyle="--", label=f"{simulator_ref.controller_name}", color=self.color_list[color_index])
            ax3.step(self.t_eval, np.append(acceleration_ref, acceleration_ref[-1]), where='post', linestyle="--", label=f"{simulator_ref.controller_name}", color=self.color_list[color_index])

            color_index += 1
            
        # Plot shadowed zone for state bounds
        if self.env.state_lbs is not None:
            ymin, _ = ax1.get_ylim()
            lower_edge = min(self.env.state_lbs[0] - self.shadow_space_wide, ymin)
            upper_edge = self.env.state_lbs[0]
            ax1.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'pos_lower_bound')
        if self.env.state_ubs is not None:
            _, ymax = ax1.get_ylim()
            lower_edge = self.env.state_ubs[0]
            upper_edge = max(self.env.state_ubs[0] + self.shadow_space_wide, ymax)
            ax1.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'pos_upper_bound')

        if self.env.state_lbs is not None:
            ymin, _ = ax2.get_ylim()
            lower_edge = min(self.env.state_lbs[1] - self.shadow_space_wide, ymin)
            upper_edge = self.env.state_lbs[1]
            ax2.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'vel_lower_bound')
        if self.env.state_ubs is not None:
            _, ymax = ax2.get_ylim()
            lower_edge = self.env.state_ubs[1]
            upper_edge = max(self.env.state_ubs[1] + self.shadow_space_wide, ymax)
            ax2.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'vel_upper_bound')
            
        if self.env.input_lbs is not None:
            ymin, _ = ax3.get_ylim()
            lower_edge = min(self.env.input_lbs-self.shadow_space_wide, ymin)
            upper_edge = self.env.input_lbs
            ax3.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'input_lower_bound')
        if self.env.input_ubs is not None:
            _, ymax = ax3.get_ylim()
            lower_edge = self.env.input_ubs
            upper_edge = max(self.env.input_ubs+self.shadow_space_wide, ymax)
            ax3.fill_between(self.t_eval, lower_edge, upper_edge, facecolor='gray', alpha=0.3, label=f'input_upper_bound')


        # Set labels and legends
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.legend()

        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.legend()

        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Input (m/s^2)")
        ax3.legend()

        if title is not None:
            fig.suptitle(title, fontsize=20, fontweight='bold', y=1.05)

        plt.tight_layout()
        plt.show()

    def display_contrast_cost2go(self, *simulators: Simulator) -> None:

        color_index = 0

        if not simulators:
            raise ValueError("No simulator references provided.")

        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        gs.update(hspace=0.4, wspace=0.3)

        # cost vs time
        ax0 = fig.add_subplot(gs[0, 0]) 
        ax0.set_title("Total Cost w.r.t. simulation time")

        # cost vs iteration
        ax1 = fig.add_subplot(gs[1, :])       
        ax1.set_title("Total Cost w.r.t. iLQR Iteration")

        ax = [ax0, ax1] 

        # Plot the reference and evaluated trajectories for each simulator
        for simulator_ref in simulators:
            if not np.all(simulator_ref.cost2go_arr):
                raise ValueError(f"Failed to get trajectory from simulator {simulator_ref.controller_name}. State trajectory list is void; please run 'run_simulation' first.")

            # Plot cost over time
            ax[0].plot(self.t_eval, np.append(simulator_ref.cost2go_arr, simulator_ref.cost2go_arr[-1]), linestyle="--", label=f"{simulator_ref.controller_name}", color=self.color_list[color_index])

            color_index += 1
        
            # Annotate total cost at initial time step
            initial_cost = simulator_ref.cost2go_arr[0]
            ax[0].annotate(
                f"Total cost: {initial_cost:.2f}",
                xy=(0, initial_cost),
                xytext=(10, initial_cost + 0.05 * initial_cost),
                arrowprops=dict(arrowstyle="->", lw=1.5),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
            )

        # Plot cost over time
        ax[0].plot(self.t_eval, np.append(self.simulator.cost2go_arr, self.simulator.cost2go_arr[-1]), label=f"{self.simulator.controller_name}", color=self.color_list[color_index])

        color_index += 1
    
        # Annotate total cost at initial time step
        initial_cost = self.simulator.cost2go_arr[0]
        ax[0].annotate(
            f"Total cost: {initial_cost:.2f}",
            xy=(0, initial_cost),
            xytext=(10, initial_cost + 0.05 * initial_cost),
            arrowprops=dict(arrowstyle="->", lw=1.5),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
        )

        # Set labels and legends
        ax[0].set_title("Total Cost w.r.t. simulation time")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Cost-to-go")
        ax[0].legend()

        # Show iLQR total cost w.r.t iterations
        if self.simulator.controller_type == 'iLQR': 

            ilqr_total_cost_list = self.simulator.controller.total_cost_list
            ilqr_total_cost_list[0] = simulator_ref.cost2go_arr[0]

            ax[1].plot(ilqr_total_cost_list, marker='o', label="iLQR")

            for simulator_ref in simulators:
                ax[1].axhline(y=simulator_ref.cost2go_arr[0], linestyle='--', color='red', label="LQR")

            ax[1].set_title("Total Cost w.r.t. iLQR Iteration")
            ax[1].set_xlabel("Iteration")
            ax[1].set_ylabel("Total Cost")
            ax[1].legend()
            ax[1].grid(True)

        plt.show()
        
    def display_phase_portrait(self):

        """
        Display RMPC results in the (x1, x2) plane, showing nominal trajectory,
        true trajectory, and invariant tube cross-sections.
        """
        
        if self.controller.type != 'RMPC':
            raise ValueError("This visualization is only supported for Tube-based RMPC controllers.")

        fig, ax = plt.subplots(figsize=(6, 6))

        # Extract true state trajectory
        x_true = self.state_traj[:, 0]
        v_true = self.state_traj[:, 1]
        
        # Extract nominal trajectory if available
        x_nom = []
        v_nom = []
        for i in range(len(self.simulator.state_pred_traj)):
            state_pred_traj_curr = self.simulator.state_pred_traj[i]
            x_nom.append(state_pred_traj_curr[0, 0])
            v_nom.append(state_pred_traj_curr[0, 1])

        # Plot nominal and true trajectories
        ax.plot(x_nom, v_nom, linestyle='--', color='black', marker='x', label='Nominal Trajectory')
        ax.plot(x_true, v_true, linestyle='-', color='blue', marker='*', label='Real Trajectory')

        # Plot red tube polygons (Ω translated to nominal)
        Omega = self.controller.Omega_tube.V
        assert Omega.shape[1] == 2, "Only 2D invariant sets are supported."
        # Compute axis-aligned bounding box
        min_bounds = np.min(self.controller.Omega_tube.V, axis=0)
        max_bounds = np.max(self.controller.Omega_tube.V, axis=0)

        for i in range(len(x_nom)):

            # Show polytopes
            center = np.array([x_nom[i], v_nom[i]])
            tube_vertices = Omega + center  # Translate tube
            if i == 0:
                patch = Polygon(tube_vertices, closed=True, edgecolor='red', facecolor='red', alpha=0.3, label='Robust Invariant Set Ω')
            else:
                patch = Polygon(tube_vertices, closed=True, edgecolor='red', facecolor='red', alpha=0.3)
            ax.add_patch(patch)

            # Show bounding box
            #bounding_box = np.array([
            #    [min_bounds[0]+x_nom[i], min_bounds[1]+v_nom[i]],
            #    [min_bounds[0]+x_nom[i], max_bounds[1]+v_nom[i]],
            #    [max_bounds[0]+x_nom[i], max_bounds[1]+v_nom[i]],
            #    [max_bounds[0]+x_nom[i], min_bounds[1]+v_nom[i]],
            #    [min_bounds[0]+x_nom[i], min_bounds[1]+v_nom[i]]
            #])  
            # Translate bounding box
            #ax.plot(bounding_box[:, 0], bounding_box[:, 1], 'r--', linewidth=2)
        
        # Show initial state and target state
        ax.plot(self.env.initial_position, self.env.initial_velocity, marker='o', color='darkorange', markersize=12, markeredgewidth=3, label='Initial State')
        ax.plot(self.env.target_position, self.env.target_velocity, marker='x', color='green', markersize=15, markeredgewidth=3, label='Target State')

        ax.set_xlabel(r"Position $p$")
        ax.set_ylabel(r"Velocity $v$")
        #ax.set_xlim(self.simulator.env.pos_lbs, self.simulator.env.pos_ubs)
        #ax.set_ylim(self.simulator.env.vel_lbs, self.simulator.env.vel_ubs)
        ax.set_title("Trajectory of RMPC in Phase Portrait with Tube Ω")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()



    def display_animation(self) -> HTML:
        
        # Instantiate the plotting
        fig, ax1 = plt.subplots(1, 1, figsize=self.figsize)
        
        # Define size of plotting
        p_max = 1.0 #max(self.position)
        p_min = -1.0 #min(self.position)
        start_extension = p_min - 0.3
        end_extension = p_max + 0.3

        p_disp_vals = np.linspace(start_extension, end_extension, 200) # generate grid mesh on p

        h_disp_vals = [float(self.env.h(p).full().flatten()[0]) for p in p_disp_vals]
        h_max = 1.0 #max(h_disp_vals)
        h_min = -1.0 #min(h_disp_vals)

        ax1.set_xlim(start_extension-0.5, end_extension+0.5)
        ax1.set_ylim(h_min-0.3, h_max+0.3)

        # Draw mountain profile curve h(p)
        ax1.plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="green")
        ax1.set_xlabel("Position p")
        ax1.set_ylabel("Height h")
        ax1.legend()

        # Mark the intial state and the target state in the plotting
        initial_h = float(self.env.h(self.env.initial_position).full().flatten()[0])
        target_h = float(self.env.h(self.env.target_position).full().flatten()[0])
        ax1.scatter([self.env.initial_position], [initial_h], color="blue", label="Start")
        #ax1.scatter([self.env.target_position], [target_h], color="orange", label="Target position")
        ax1.plot(self.env.target_position, target_h, marker='x', color='red', markersize=10, markeredgewidth=3, label='Target')

        if self.controller.type == 'LQR' and not np.allclose(self.controller.state_lin, self.controller.target_state):
            lin_position = self.controller.state_lin[0]
            lin_h = float(self.env.h(lin_position).full().flatten()[0])
            ax1.plot(lin_position, lin_h, marker='v', color='orange', markersize=7, markeredgewidth=3, label='Linearization Point')

        ax1.legend()


        # Setting simplyfied car model as rectangle, and update the plotting to display the animation
        car_height = self.car_length / 2
        car = Rectangle((0, 0), self.car_length, car_height, color="black")
        ax1.add_patch(car)

        def update(frame):
            # Get current position and attitude of car
            current_position = self.position[frame]
            current_theta = float(self.env.theta(current_position).full().flatten()[0])

            # Update position and attitude of car
            car.set_xy((current_position - self.car_length / 2, float(self.env.h(current_position - self.car_length / 2).full().flatten()[0])))
            car.angle = np.degrees(current_theta)  # rad to deg

        # Instantiate animation
        anim = FuncAnimation(fig, update, frames=len(self.t_eval), interval=1000 / self.refresh_rate, repeat=False)

        plt.close(fig)

        return HTML(anim.to_jshtml())
    
    def display_contrast_animation(self, *simulators) -> HTML:

        # Instantiate the plotting
        num_plots = len(simulators) + 1
        fig, axes = plt.subplots(num_plots, 1, figsize=(self.figsize[0], self.figsize[1] * num_plots), sharex=True)

        # Define size of plotting
        p_max = 1.0 #max(max(sim.get_trajectories()[0][:, 0]) for sim in simulators)
        p_min = -1.0 #min(min(sim.get_trajectories()[0][:, 0]) for sim in simulators)
        start_extension = p_min - 0.3
        end_extension = p_max + 0.3

        p_disp_vals = np.linspace(start_extension, end_extension, 200)  # generate grid mesh on p

        h_disp_vals = [float(self.env.h(p).full().flatten()[0]) for p in p_disp_vals]
        h_max = 1.0 #max(h_disp_vals)
        h_min = -1.0 #min(h_disp_vals)
        axes[0].set_xlim(start_extension - 0.5, end_extension + 0.5)
        axes[0].set_ylim(h_min - 0.3, h_max + 0.3)
        axes[0].plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="green")
        axes[0].set_xlabel("Position p")
        axes[0].set_ylabel("Height h")

        for ax, sim in zip(axes[1:], simulators):

            h_disp_vals = [float(sim.env.h(p).full().flatten()[0]) for p in p_disp_vals]
            ax.set_xlim(start_extension - 0.5, end_extension + 0.5)
            ax.set_ylim(h_min - 0.3, h_max + 0.3)
            ax.plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="green")
            ax.set_xlabel("Position p")
            ax.set_ylabel("Height h")


        # Mark the initial state and the target state in the plotting
        initial_h = float(self.env.h(self.env.initial_position).full().flatten()[0])
        target_h = float(self.env.h(self.env.target_position).full().flatten()[0])
        axes[0].scatter([self.env.initial_position], [initial_h], color="blue", label="Start")
        #axes[0].scatter([self.env.target_position], [target_h], color="orange", label="Target position")
        axes[0].plot(self.env.target_position, target_h, marker='x', color='red', markersize=10, markeredgewidth=3, label='Target')
        axes[0].legend()

        for ax, sim in zip(axes[1:], simulators):

            initial_h = float(sim.env.h(self.env.initial_position).full().flatten()[0])
            target_h = float(sim.env.h(self.env.target_position).full().flatten()[0])

            ax.scatter([sim.env.initial_position], [initial_h], color="blue", label="Start")

            #ax.scatter([sim.env.target_position], [target_h], color="orange", label="Target position")
            ax.plot(sim.env.target_position, target_h, marker='x', color='red', markersize=10, markeredgewidth=3, label='Target')


            if sim.controller.type == 'LQR' and not np.allclose(sim.controller.state_lin, sim.controller.target_state):
                lin_position = sim.controller.state_lin[0]
                lin_h = float(sim.env.h(lin_position).full().flatten()[0])
                ax.plot(lin_position, lin_h, marker='v', color='orange', markersize=7, markeredgewidth=3, label='Linearization Point of LQR')

            ax.legend()


        # Create car objects for each simulator
        car_objects = {}
        colors = self.color_list[:len(simulators)]
        car_height = self.car_length / 2

        car_self = Rectangle((0, 0), self.car_length, car_height, color=self.color)
        axes[0].add_patch(car_self)
        axes[0].set_title(f"{self.simulator.controller_name}")

        for ax, sim, color in zip(axes[1:], simulators, colors):
            car = Rectangle((0, 0), self.car_length, car_height, color=color)
            ax.add_patch(car)
            ax.set_title(f"{sim.controller_name}")
            car_objects[sim] = car


        def update(frame):
            # Update car for self
            current_position = self.position[frame]
            current_theta = float(self.env.theta(current_position).full().flatten()[0])
            car_self.set_xy((current_position - self.car_length / 2, float(self.env.h(current_position - self.car_length / 2).full().flatten()[0])))
            car_self.angle = np.degrees(current_theta)  # rad to deg

            for sim, car in car_objects.items():
                current_position = sim.get_trajectories()[0][:, 0][frame]
                current_theta = float(sim.env.theta(current_position).full().flatten()[0])
                car.set_xy((current_position - self.car_length / 2, float(sim.env.h(current_position - self.car_length / 2).full().flatten()[0])))
                car.angle = np.degrees(current_theta)  # rad to deg

        # Instantiate animation
        anim = FuncAnimation(fig, update, frames=len(self.t_eval), interval=1000 / self.refresh_rate, repeat=False)

        plt.close(fig)

        return HTML(anim.to_jshtml())
    
    def display_contrast_animation_same(self, *simulators) -> HTML:
        import matplotlib.patches as mpatches

        custom_handles = []

        # Setup figure
        fig, ax = plt.subplots(1, 1, figsize=(self.figsize[0], self.figsize[1]))

        # Plot mountain profile
        p_max, p_min = 1.0, -1.0
        start_extension, end_extension = p_min - 0.3, p_max + 0.3
        p_disp_vals = np.linspace(start_extension, end_extension, 200)
        h_disp_vals = [float(self.env.h(p).full().flatten()[0]) for p in p_disp_vals]

        ax.set_xlim(start_extension - 0.5, end_extension + 0.5)
        ax.set_ylim(-1.3, 1.3)
        profile_plot, = ax.plot(p_disp_vals, h_disp_vals, label="Mountain profile h(p)", color="green")
        custom_handles.append(profile_plot)

        ax.set_xlabel("Position p")
        ax.set_ylabel("Height h")

        # Start & Target markers
        initial_h = float(self.env.h(self.env.initial_position).full().flatten()[0])
        target_h = float(self.env.h(self.env.target_position).full().flatten()[0])
        start_scatter = ax.scatter([self.env.initial_position], [initial_h], color="blue", label="Start")
        custom_handles.append(start_scatter)
        target_cross = ax.plot(self.env.target_position, target_h, marker='x', color='red', markersize=10, markeredgewidth=3, label='Target')[0]
        custom_handles.append(target_cross)

        for sim in simulators:

            if sim.controller.type == 'LQR' and not np.allclose(sim.controller.state_lin, sim.controller.target_state):
                lin_position = sim.controller.state_lin[0]
                lin_h = float(sim.env.h(lin_position).full().flatten()[0])
                lin_point = ax.plot(lin_position, lin_h, marker='v', color='orange', markersize=7, markeredgewidth=3, label='Linearization Point of LQR')[0]
                custom_handles.append(lin_point)

        # Car setup
        car_objects = {}
        colors = [self.color] + self.color_list[:len(simulators)]
        car_height = self.car_length / 2

        # Self car
        car_self = Rectangle((0, 0), self.car_length, car_height,
                            edgecolor=self.color, facecolor='none', linewidth=2)
        ax.add_patch(car_self)
        car_objects[self] = car_self

        # Simulators' cars
        for sim, color in zip(simulators, colors[1:]):
            car = Rectangle((0, 0), self.car_length, car_height,
                            edgecolor=color, facecolor='none', linewidth=2)
            ax.add_patch(car)
            car_objects[sim] = car

        # Title using correct controller names
        controller_names = " vs. ".join(
            [self.simulator.controller_name] + [sim.controller_name for sim in simulators]
        )
        ax.set_title(controller_names)

        # Legend

        car_legend_handles = [
            mpatches.Patch(edgecolor=self.color, facecolor='none', linewidth=2, label=self.simulator.controller_name)
        ]
        for sim, color in zip(simulators, colors[1:]):
            car_legend_handles.append(
                mpatches.Patch(edgecolor=color, facecolor='none', linewidth=2, label=sim.controller_name)
            )

        ax.legend(handles=custom_handles + car_legend_handles, loc='best')

        # Animation update function
        def update(frame):
            # Self
            current_position = self.position[frame]
            current_theta = float(self.env.theta(current_position).full().flatten()[0])
            y_base = float(self.env.h(current_position - self.car_length / 2).full().flatten()[0])
            car_self.set_xy((current_position - self.car_length / 2, y_base))
            car_self.angle = np.degrees(current_theta)

            # Simulators
            for sim, car in car_objects.items():
                if sim is self:
                    continue
                current_position = sim.get_trajectories()[0][:, 0][frame]
                current_theta = float(sim.env.theta(current_position).full().flatten()[0])
                y_base = float(sim.env.h(current_position - self.car_length / 2).full().flatten()[0])
                car.set_xy((current_position - self.car_length / 2, y_base))
                car.angle = np.degrees(current_theta)

        # Animate
        anim = FuncAnimation(fig, update, frames=len(self.t_eval),
                            interval=1000 / self.refresh_rate, repeat=False)
        
        plt.close(fig)
        
        return HTML(anim.to_jshtml())







# Actions:
# add a dataclass to set up all internal & public variable in class (https://docs.python.org/3/library/dataclasses.html)

