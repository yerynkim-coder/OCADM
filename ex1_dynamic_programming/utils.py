import numpy as np
import casadi as ca
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML


class Env:
    def __init__(self, case, init, target, symbolic_h, symbolic_theta):
        self.case = case  # 1, 2, 3, 4
        self.initial_position = init
        self.target_position = target
        self.constraint = None

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
        
        # Define system dynamics
        self.dynamics_function = dynamics_eq(env.theta)

        # parameter for test
        self.__a = 1
        self.__t_0 = 0
        self.__t_terminal = 5
        self.__dt = 0.01
        self.__init_state = [0, 0] 

    def test_dynamics(self):

        # Solve the dynamics with test-input
        N = int((self.__t_terminal - self.__t_0) / self.__dt)

        t_span = [self.__t_0, self.__t_terminal]
        t_eval = np.linspace(self.__t_0, self.__t_terminal, N)
        
        sim_dynamics = lambda t, state: self.dynamics.dynamics_function(state, [self.__a]).full().flatten()
        solution = solve_ivp(sim_dynamics, t_span, self.init_state_test, t_eval=t_eval)

        # Display p (left), v (middle), a (right)
        a_values = [self.__a] * len(t_eval)

        _, ax = plt.subplots(1, 3, figsize=(12, 3))

        # p(t)
        ax[0].plot(solution.t, solution.y[0], label="p(t)")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Position p(t)")
        ax[0].legend()
        ax[0].set_title("Position over Time")

        # v(t) 
        ax[1].plot(solution.t, solution.y[1], label="v(t)")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Velocity v(t)")
        ax[1].legend()
        ax[1].set_title("Velocity over Time")

        # a(t)
        ax[2].plot(solution.t, a_values, label="a(t)")
        ax[2].set_xlabel("Time")
        ax[2].set_ylabel("Acceleration a(t)")
        ax[2].legend()
        ax[2].set_title("Acceleration over Time")

        plt.tight_layout()
        plt.show()


# TODO: derive from a interface class
class PIDController:
    def __init__(self, kp=0.1, ki=0.1, kd=0.05, target_position=5.0):
        
        # TODO: parameter tuning
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.target_position = target_position
        self.integral = 0
        self.prev_error = 0

    def compute_control(self, current_position, dt):

        # Calculate position error
        error = self.target_position - current_position

        # Calculate integral and difference
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt

        # Update old value
        self.prev_error = error

        # Use PID law to calculate command (a)
        a = self.kp * error + self.ki * self.integral + self.kd * derivative
        return a


class Simulator:
    def __init__(self, dynamics, controller, env):

        self.dynamics = dynamics
        self.controller = controller
        self.env = env
        
        # Define timeline
        self.t_0 = 0
        self.t_terminal = 20
        self.dt = 0.05
        self.t_eval = np.linspace(self.t_0, self.t_terminal, int((self.t_terminal - self.t_0) / self.dt))
        
        # Initialize recording list for p / v / a
        self.positions = []
        self.velocities = []
        self.accelerations = []
        
        # Define setting of animation
        self.refresh_rate = 30

    def run_simulation(self):

        current_state = [self.env.initial_position, 0]

        for current_time in self.t_eval:

            # Get current state, and call controller to calculate input
            current_speed = current_state[1]
            a = self.controller.compute_control(current_speed, self.dt)

            # Do one-step simulation
            t_span = [current_time, current_time + self.dt]
            sim_dynamics = lambda t, state: self.dynamics.dynamics_function(state, [a]).full().flatten()
            solution = solve_ivp(sim_dynamics, t_span, current_state, t_eval=[current_time + self.dt])
            current_state = solution.y[:, -1]

            # Log the results
            self.positions.append(current_state[0])
            self.velocities.append(current_state[1])
            self.accelerations.append(a)

    '''
    def display_animation(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
        ax1.set_xlim(self.dynamics.t_0, self.dynamics.t_terminal)
        ax1.set_ylim(0, self.env.target)
        ax2.set_xlim(self.dynamics.t_0, self.dynamics.t_terminal)
        ax2.set_ylim(-1, 1)
        
        
        def update(frame):

            h_values = [float(self.env.h(p).full().flatten()[0]) for p in self.positions[:frame + 1]]
            theta_values = [float(self.env.theta(p).full().flatten()[0]) for p in self.positions[:frame + 1]]

            # Update the figures
            ax1.clear()
            ax1.plot(self.t_eval[:frame + 1], self.positions[:frame + 1], label="Position", color="blue")
            ax1.plot(self.t_eval[:frame + 1], self.velocities[:frame + 1], label="Velocity", color="green")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Position / Velocity")
            ax1.legend()

            ax2.clear()
            ax2.plot(self.positions[:frame + 1], h_values, label="h(p)", color="green")
            ax2.plot(self.positions[:frame + 1], theta_values, label=r"$\theta(p)$", color="blue")
            ax2.set_xlabel("Position p")
            ax2.set_ylabel("h / theta")
            ax2.legend()

        # Instantiation
        anim = FuncAnimation(fig, update, frames=len(self.t_eval), interval=1000 / self.refresh_rate, repeat=False)

        # Display animation
        ??????
    '''

    def display_final_results(self):

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot p / v / a over time t
        ax.plot(self.t_eval, self.positions, label="Position p(t)", color="blue")
        ax.plot(self.t_eval, self.velocities, label="Velocity v(t)", color="green")
        ax.plot(self.t_eval, self.accelerations, label="Acceleration a(t)", color="red")
        ax.set_xlabel("Time")
        ax.set_ylabel("Position / Velocity / Acceleration")
        ax.legend()
        ax.set_title("Position, Velocity, and Acceleration over Time")

        plt.tight_layout()
        plt.show()


'''
class Controller(ABC):

    def __init__(self):
        self.input = None
        self._target = None

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = value
    
    @abstractmethod
    def test_dynamics():
        pass
    
class PIDController(Controller):
    def __init__(self, kp, ki, kd):
        super().__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def control(self, current_value):
        if self.target is None:
            raise ValueError("Target value is not set.")
        
        error = self.target - current_value
        self.integral += error
        derivative = error - self.previous_error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

    def set_target(self, new_target):
        self.target = new_target  # call @target.setter
'''

    