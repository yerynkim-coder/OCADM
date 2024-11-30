import numpy as np
import casadi as ca
import scipy.linalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML

# Add a tutorial to show how to install acados and set interface to python
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel




class Env:
    def __init__(self, case, init_state, target_state, symbolic_h, symbolic_theta):

        self.case = case  # 1, 2, 3, 4

        self.initial_position = init_state[0]
        self.initial_velocity = init_state[1]
        self.init_state = np.array([self.initial_position, self.initial_velocity])

        self.target_position = target_state[0]
        self.target_velocity = target_state[1]
        self.target_state = np.array([self.target_position, self.target_velocity])


        # Define argument p as CasADi symbolic parameters
        p = ca.SX.sym("p")

        # Set functions h(p) and theta(p) based on symbolic parameters
        self.h = ca.Function("h", [p], [symbolic_h(p, case)])
        self.theta = symbolic_theta(self.h)

    def test_env(self, p_vals):
        
        # Calculate values of h and theta on grid mesh
        h_vals = [float(self.h(p)) for p in p_vals]
        theta_vals = [float(self.theta(p)) for p in p_vals]
        
        # Display curve theta(p) (left), h(p) (right)
        _, ax = plt.subplots(1, 2, figsize=(12, 3))
    
        # theta(p)
        ax[0].plot(p_vals, theta_vals, label=r"$\theta(p)$", color='blue')
        ax[0].set_xlabel("p")
        ax[0].set_ylabel(r"$\theta$")
        ax[0].set_ylim(-1.5, 1.5)  
        ax[0].set_title(r"$\theta$($p$)")
        ax[0].legend()

        # h(p)
        ax[1].plot(p_vals, h_vals, label="h(p)", color='green')
        ax[1].set_xlabel("p")
        ax[1].set_ylabel("h")
        ax[1].set_title("h(p)")
        ax[1].legend()
        plt.show()




class Dynamics:
    def __init__(self, state_names, input_names, dynamics_eq, env):

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

    def get_linearized_AB_discrete(self, current_state, current_input, dt):
        '''use current state, and current input to calculate the linearized state transfer matrix A_lin and input matrix B_lin'''

        # Evaluate linearized A and B around set point
        A_lin = np.array(self.A_func(current_state, current_input))
        B_lin = np.array(self.B_func(current_state, current_input))

        # Transform into discrete dynamics
        A_lin = A_lin * dt + np.eye(self.dim_states)
        B_lin = B_lin * dt
 
        return A_lin, B_lin
    
    def compute_equilibrium_input(self, state):

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
    
    def one_step_forward(self, current_state, current_input, dt):
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




class BaseController(ABC):
    def __init__(self, name, env, dynamics, freq, verbose=False):

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
    def setup(self):
        """
        Initialize necessary matrices or parameters for the controller.
        """
        pass

    @abstractmethod
    def compute_action(self, current_state, current_time):
        """
        Compute control action based on the current state and time.
        """
        pass

# Derived class for LQR Controller
class LQRController(BaseController):
    def __init__(self, env, dynamics, Q, R, freq, verbose=False):

        super().__init__('LQR', env, dynamics, freq, verbose)

        self.Q = Q
        self.R = R

        self.A = None  # State transfer matrix
        self.B = None  # Input matrix
        self.K = None  # LQR gain matrix
        
        # need no external objects for setup, can directly call the setup function here
        self.setup()

    def setup(self):
        # Linearize dynamics at equilibrium
        state_eq = np.zeros(self.dim_states)
        input_eq = np.zeros((1,))
        self.A, self.B = self.dynamics.get_linearized_AB_discrete(
            current_state=state_eq, current_input=input_eq, dt=self.dt
        )

        # Solve DARE to compute gain matrix
        P = scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ (self.B.T @ P)

        if self.verbose:
            print(f"LQR Gain Matrix K: {self.K}")

    def compute_action(self, current_state, current_time):
        if self.K is None:
            raise ValueError("LQR gain matrix K is not computed. Call setup() first.")

        # Compute state error
        det_x = current_state - self.target_state

        # Apply control law
        u = -self.K @ det_x
        return u

# Derived class for iLQR Controller
class iLQRController(BaseController):
    def __init__(self, env, dynamics, Q, R, Qf, freq, max_iter=30, tol=2e-2, verbose=False):

        super().__init__('iLQR', env, dynamics, freq, verbose)

        self.Q = Q
        self.R = R
        self.Qf = Qf  # Terminal cost matrix

        self.max_iter = max_iter
        self.tol = tol

        self.K_k_arr = None
        self.u_kff_arr = None

    def setup(self, input_traj):
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

    def compute_action(self, current_state, current_time):
        if self.K_k_arr is None or self.u_kff_arr is None:
            raise ValueError("iLQR parameters are not computed. Call setup() first.")

        u = self.u_kff_arr[current_time] + self.K_k_arr[:, current_time].T @ current_state
        return u

# Derived class for MPC Controller
class MPCController(BaseController):
    def __init__(self, env, dynamics, Q, R, Qf, freq, N, verbose=False):
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

        super().__init__('MPC', env, dynamics, freq, verbose)

        self.Q = Q
        self.R = R
        self.Qf = Qf

        self.N = N  # Prediction horizon

        self.freq = freq
        self.dt = 1 / self.freq  # Time step

        self.ocp = None
        self.solver = None

        self.setup()

    def setup(self):
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

        # Constraints # TODO: specify from env/dynamics
        ocp.constraints.x0 = self.init_state  # Initial state
        ocp.constraints.lbx = np.array([-1e6, -1e6])
        ocp.constraints.ubx = np.array([1e6, 1e6])
        ocp.constraints.idxbx = np.arange(self.dim_states)
        ocp.constraints.lbu = np.array([-1e6]) * self.dim_inputs  # Lower bound on input
        ocp.constraints.ubu = np.array([1e6]) * self.dim_inputs  # Upper bound on input
        ocp.constraints.idxbu = np.array(self.dim_inputs)  # Input index for constraints

        # Set up Acados solver
        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file=f"{model.name}.json")

        if self.verbose:
            print("MPC setup with Acados completed.")

    def compute_action(self, current_state, current_time):
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
    def __init__(self, dynamics, controller, env, dt, t_terminal, verbose=False):

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

    def reset_counter(self):
        self.counter = 0
    
    def run_simulation(self):
        
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

    def get_trajectories(self):
        '''Get state and input trajectories, return in ndarray form '''

        # Transform state and input traj from list to ndarray
        state_traj = np.array(self.state_traj)
        input_traj = np.array(self.input_traj)

        return state_traj, input_traj




class Visualizer:

    def __init__(self, simulator):

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

    def display_final_results(self):

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

    def display_contrast(self, simulator_ref):
        
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

    def display_animation(self):
        
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
# derive iLQR form LQR
# split up teh controlelr and simulator
# update the implementation of discritization with matrix exponential (scipy)
# add typing for each function argument and output (https://docs.python.org/3/library/typing.html)
# add property decorator to limti the charactor of argument like pdf (https://stackoverflow.com/questions/17330160/how-does-the-property-decorator-work-in-python)
# add a dataclass to set up all internal & public variable in class (https://docs.python.org/3/library/dataclasses.html)


# Next steps:
# finish PID
# implement ADP (without constraint)
