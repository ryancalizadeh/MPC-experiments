import numpy as np
import cvxpy as cp
from CartPole import CartPole
from scipy.linalg import solve_discrete_are

class MPC:
    def __init__(self, cart: CartPole, N: int = 30, x0: np.ndarray = np.zeros(4)):
        self.N = N  # Prediction horizon
        self.cart = cart

        # Precompute matrices for the optimization problem
        self.A_bar = np.zeros((4 * (N + 1), 4))
        for i in range(N + 1):
            self.A_bar[4*i:4*(i+1), :] = np.linalg.matrix_power(cart.A, i)

        self.B_bar = np.zeros((4 * (N + 1), N))
        for i in range(1, N + 1):
            for j in range(i):
                self.B_bar[4*i:4*(i+1), j] = (np.linalg.matrix_power(cart.A, i - j - 1) @ cart.B).flatten()
        
        # print("A_bar:", self.A_bar)
        # print("B_bar:", self.B_bar)

        # Input constraint matrix
        self.G_bar = np.kron(np.eye(N), np.array([[1], [-1]]))
        self.g_bar = np.kron(np.ones((2 * N, 1)), cart.saturation_limit)
        # print("G_bar:", self.G_bar)
        # print("g_bar:", self.g_bar)

        # Terminal state constraint region
        self.F_final = np.kron(np.eye(4), np.array([[1], [-1]]))
        self.f_final = np.ones((8, 1)) * 100.0
        # print("F_final:", self.F_final)
        # print("f_final:", self.f_final)

        self.F_bar = np.zeros((8 * (N + 1), 4 * (N + 1)))
        self.F_bar[-8:, -4:] = self.F_final

        self.f_bar = np.zeros((8 * (N + 1), 1))
        self.f_bar[-8:] = self.f_final
        # print("F_bar:", self.F_bar)
        # print("f_bar:", self.f_bar)

        # Cost matrices
        self.Q_bar = np.kron(np.eye(N+1), cart.Q)
        self.R_bar = np.kron(np.eye(N), cart.R)

        # Add terminal cost
        Qf = solve_discrete_are(cart.A, cart.B, cart.Q, cart.R)
        self.Q_bar[-4:, -4:] = Qf
        # print("Q_bar:", self.Q_bar)
        # print("R_bar:", self.R_bar)

        # Warm start variable
        self.u_var = cp.Variable((N, 1))
        self.solve(x0, cart, max_steps=100000)

    def solve(self, x:np.ndarray, cart: CartPole, max_steps: int = 1000) -> np.ndarray:

        constraints = []

        # Define the constraints
        constraints += [self.F_bar @ self.B_bar @ self.u_var <= self.f_bar - self.F_bar @ self.A_bar @ x.reshape(-1, 1)]
        constraints += [self.G_bar @ self.u_var <= self.g_bar]

        # Define the cost function
        cost = 0

        cost += 2 * x.reshape(-1, 1).T @ self.A_bar.T @ self.Q_bar @ self.B_bar @ self.u_var
        cost += cp.quad_form(self.u_var, cp.psd_wrap(self.B_bar.T @ self.Q_bar @ self.B_bar + self.R_bar))

        # Define and solve the optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, max_iter=max_steps, warm_start=True)

        # Return the first control input
        if self.u_var.value is not None:
            return self.u_var.value[0]
        else:
            print("MPC problem is infeasible.")
            return np.array([0.0])  # Fallback control input if problem is infeasible
