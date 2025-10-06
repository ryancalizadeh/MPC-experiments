import numpy as np
import scipy.linalg as la

class CartPole:
    '''
    x: state vector [x, theta, x_dot, theta_dot]
    u: input vector [F]
    '''

    def __init__(self, m1: float = 1, m2: float = 5, g: float = 9.81, l: float = 2, T: float = 0.001, saturation_limit: float = 10.0):

        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.l = l
        self.T = T
        self.saturation_limit = saturation_limit

        self.A_cont = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, (m2 * g) / m1, 0, 0],
            [0, g * (m1 + m2) / (l * m1), 0, 0]
        ])

        self.B_cont = np.array([
            [0],
            [0],
            [1 / m1],
            [1 / (l * m1)]
        ])

        AB = np.block([
            [self.A_cont, self.B_cont],
            [np.zeros((1, 4)), 0]
        ])
        expAB = la.expm(AB * self.T)
        self.A = expAB[:4, :4]
        self.B = expAB[:4, 4:5]

        self.Q = np.diag([10, 100, 1, 1])  # State cost
        self.R = np.array([[0.01]])          # Input cost
    
    def linear_plant(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
        x: state vector [x, theta, x_dot, theta_dot]
        '''

        x_next = self.A @ x + self.B @ u

        return x_next
    
    def nonlinear_plant_cont(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
        x: state vector [x, theta, x_dot, theta_dot]
        u: input vector [F]
        '''

        x_pos, theta, x_dot, theta_dot = x
        F = u[0]
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        x_ddot = (F + self.m2 * sin_theta * (-self.l * theta_dot**2 + self.g * cos_theta)) / (self.m1 + self.m2 * sin_theta**2)
        theta_ddot = (F * cos_theta - self.m2 * self.l * theta_dot**2 * cos_theta * sin_theta + (self.m1 + self.m2) * self.g * sin_theta) / (self.l * (self.m1 + self.m2 * sin_theta**2))

        f = np.array([
            x_dot,
            theta_dot,
            x_ddot,
            theta_ddot
        ])

        return f
    
    def rk4_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        '''
        x: state vector [x, theta, x_dot, theta_dot]
        u: input vector [F]
        '''

        T = self.T
        k1 = self.nonlinear_plant_cont(x, u)
        k2 = self.nonlinear_plant_cont(x + 0.5 * T * k1, u)
        k3 = self.nonlinear_plant_cont(x + 0.5 * T * k2, u)
        k4 = self.nonlinear_plant_cont(x + T * k3, u)

        x_next = x + (T / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next




