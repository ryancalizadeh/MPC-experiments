import control as ct
import numpy as np
from CartPole import CartPole

def lqr(x: np.ndarray, cart: CartPole) -> np.ndarray:

    # Compute the LQR gain
    K, S, E = ct.dlqr(cart.A, cart.B, cart.Q, cart.R)

    # Compute the control input
    u = -K @ x.T  # State feedback control

    return u