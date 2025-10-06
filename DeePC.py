import numpy as np
import cvxpy as cp
from CartPole import CartPole
import lqr

class DeePC:
    def __init__(
            self,
            cart: CartPole,
            data_u: np.ndarray,
            data_x: np.ndarray,
            T_ini: int = 1,
            N: int = 50,
            x0: np.ndarray = np.zeros(4)):
        """
        data_u.shape = (m, T)
        data_x.shape = (n, T)
        """

        self.k = 0
        self.N = N
        self.T_ini = T_ini
        self.cart = cart
        self.T = data_u.shape[1]

        self.x_ini = np.zeros((4, T_ini))
        self.u_ini = np.zeros((1, T_ini))

        # Construct Hankel matrices
        U = self.block_hankel(data_u, T_ini + N)
        X = self.block_hankel(data_x, T_ini + N)

        print(U.shape)
        print(X.shape)

        # Print row rank of U and X
        print("Row rank of U:", np.linalg.matrix_rank(U))
        print("Row rank of X:", np.linalg.matrix_rank(X))

        self.U_p = U[:T_ini, :]
        self.U_f = U[T_ini:, :]
        self.X_p = X[:4*T_ini, :]
        self.X_f = X[4*T_ini:, :]

        print("Shape of U_p:", self.U_p.shape)
        print("Shape of U_f:", self.U_f.shape)
        print("Shape of X_p:", self.X_p.shape)
        print("Shape of X_f:", self.X_f.shape)

        print("Row rank of U_p:", np.linalg.matrix_rank(self.U_p))
        print("Row rank of U_f:", np.linalg.matrix_rank(self.U_f))
        print("Row rank of X_p:", np.linalg.matrix_rank(self.X_p))
        print("Row rank of X_f:", np.linalg.matrix_rank(self.X_f))
        #exit()

        # Input constraint matrix
        self.G_bar = np.kron(np.eye(N), np.array([[1], [-1]]))
        self.g_bar = np.kron(np.ones((2 * N, 1)), cart.saturation_limit)
        # print("G_bar:", self.G_bar)
        # print("g_bar:", self.g_bar)

        # Cost matrices
        self.Q_bar = np.kron(np.eye(N), cart.Q)
        self.R_bar = np.kron(np.eye(N), cart.R)

        print(self.X_f.shape)
        print(self.U_f.shape)

        self.cost_matrix = self.X_f.T @ self.Q_bar @ self.X_f + self.U_f.T @ self.R_bar @ self.U_f

        self.lambda_s = 1e5
        self.lambda_g = 200
        
        # Warm start variable
        self.g_var = cp.Variable((self.T-self.T_ini-N+1, 1))
        self.sigma = cp.Variable((4 * self.T_ini, 1))

    def solve(self, x:np.ndarray, cart: CartPole) -> np.ndarray:
        self.x_ini[:, :-1] = self.x_ini[:, 1:]
        self.x_ini[:, -1] = x

        constraints = []

        # Define the constraints
        constraints += [self.X_p @ self.g_var == self.x_ini.reshape(-1, 1) + self.sigma]
        constraints += [self.U_p @ self.g_var == self.u_ini.reshape(-1, 1)]
        constraints += [self.G_bar @ self.U_f @ self.g_var <= self.g_bar]

        # Define the cost function
        cost = 0
        cost += cp.quad_form(self.g_var, cp.psd_wrap(self.cost_matrix))
        cost += self.lambda_s * cp.sum_squares(self.sigma)
        cost += self.lambda_g * cp.sum_squares(self.g_var)

        # Define and solve the optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP)

        # Return the first control input
        if self.g_var.value is not None and self.sigma.value is not None:
            u = (self.U_f @ self.g_var.value)[0]
            print(f"DeePC control: {u}")
            print(F"Sigma norm: {np.linalg.norm(self.sigma.value, 1)}")

            # Update initial conditions for next iteration
            self.u_ini[:, :-1] = self.u_ini[:, 1:]
            self.u_ini[:, -1] = u

            return u
        else:
            print("DeePC problem is infeasible.")

            # Update initial conditions for next iteration
            self.u_ini[:, :-1] = self.u_ini[:, 1:]
            self.u_ini[:, -1] = 0.0

            return np.array([0.0])  # Fallback control input if problem is infeasible
        

    def block_hankel(self, blocks: np.ndarray, L: int) -> np.ndarray:
        """
        Construct a block Hankel matrix from an array of vectors.

        Parameters
        ----------
        blocks : np.ndarray
            Array of vectors [v1, v2, ..., v_{T}]
            blocks.shape = (p, T)
        L : int
            Number of block rows.

        Returns
        -------
        H : np.ndarray
            Block Hankel matrix of shape (L*p, T-L+1).
        """

        p = blocks.shape[0]  # Dimension of each vector
        T = blocks.shape[1]  # Total number of vectors

        # --- Construct Hankel matrix ---
        H = np.zeros((L * p, T - L + 1))
        for i in range(L):
            H[i*p:(i+1)*p, :] = blocks[:, i:i+T-L+1]

        return H
