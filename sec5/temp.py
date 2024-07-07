import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class InvertedPendulumEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is an environment for an inverted pendulum controlled by a step input.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(InvertedPendulumEnv, self).__init__()

        # Define action and observation space
        # Actions are continuous and bounded
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        
        # Observation space: [angle, angular velocity]
        self.observation_space = spaces.Box(low=np.array([-np.pi, -10]), high=np.array([np.pi, 10]), dtype=np.float32)
        
        # Constants for the inverted pendulum
        self.g = 9.81   # Acceleration due to gravity (m/s^2)
        self.L = 1.0    # Length of the pendulum (m)
        self.m = 0.1    # Mass of the pendulum (kg)
        self.b = 0.05   # Damping coefficient (kg*m^2/s)
        
        # Time step for simulation
        self.dt = 0.02
        
        # Initial conditions
        self.state = None
        self.reset()

    def inverted_pendulum_ode(self, t, y, u):
        theta, omega = y
        dtheta_dt = omega
        domega_dt = (u - self.b * omega - self.m * self.g * self.L * np.sin(theta)) / (self.m * self.L ** 2)
        return [dtheta_dt, domega_dt]

    def step(self, action):
        u = action[0]
        y0 = self.state
        
        # Integrate ODE
        sol = solve_ivp(self.inverted_pendulum_ode, [0, self.dt], y0, args=(u,), t_eval=[self.dt])
        self.state = sol.y[:, -1]
        
        # Calculate reward (e.g., penalize large angles and velocities)
        theta, omega = self.state
        reward = -(theta**2 + 0.1 * omega**2 + 0.001 * (u**2))
        
        # Check if the pendulum has fallen
        done = bool(np.abs(theta) > np.pi/2)
        
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = np.array([0.1, 0.0], dtype=np.float32)  # Small initial angle and zero initial angular velocity
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human'):
        # Simple rendering using matplotlib
        theta = self.state[0]
        
        x = self.L * np.sin(theta)
        y = -self.L * np.cos(theta)
        
        plt.figure(figsize=(5, 5))
        plt.plot([0, x], [0, y], 'r-')
        plt.plot(x, y, 'bo')
        plt.xlim(-self.L, self.L)
        plt.ylim(-self.L, self.L)
        plt.grid()
        plt.title('Inverted Pendulum')
        plt.show()

    def close(self):
        pass

# Create the environment
env = InvertedPendulumEnv()

# Test the environment
observation = env.reset()
for _ in range(200):
    action = env.action_space.sample()  # Random action
    observation, reward, done, info = env.step(action)
    if done:
        print("Pendulum fell! Resetting environment.")
        observation = env.reset()
    env.render()
    env.close()
