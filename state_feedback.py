import numpy as np
from CartPole import CartPole

def state_feedback(x: np.ndarray, cart: CartPole) -> np.ndarray:
    K = np.array([[4, 500, 2, 1]])  # State feedback gain
    u = - K @ x.T  # Simple state feedback control
    print(np.abs(np.linalg.eigvals(cart.A - cart.B @ K)))
    exit()
    return u