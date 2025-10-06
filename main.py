import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from CartPole import CartPole
from mpc import MPC
from DeePC import DeePC
from state_feedback import state_feedback
from lqr import lqr

# Name of the GIF file to save the animation
gif_name = "state_feedback.gif"
# gif_name = "lqr.gif"

# Dynamic parameters
m1 = 0.5
m2 = 0.5
g = 9.81
l = 2
saturation_limit = 10.0

# simulation parameters
T = 0.005
T_sim = 5.0
steps = int(np.floor(T_sim/T))

cart = CartPole(m1=m1, m2=m2, g=g, l=l, T=T, saturation_limit=saturation_limit)

# Initial state
xs = np.zeros((4, steps))
xs[:, 0] = np.array([0, 0.1, 0, 0])  # Slightly perturbed from upright position
us = np.zeros((1, steps - 1))

# Control specs
data_x = np.load("x_data.npy")
data_u = np.load("u_data.npy")
mpc = MPC(cart, x0 = xs[:, 0])
deepc = DeePC(cart, data_u, data_x)
# control = lqr 
control = state_feedback
# control = mpc.solve
# control = deepc.solve

def main():
    for k in range(steps - 1):
        # Simulate the nonlinear plant
        us[:, k] = np.clip(control(xs[:, k], cart), -saturation_limit, saturation_limit)
        u = us[:, k]
        
        # if k in range(150, 200):
        #     u += 5.0
        
        # if k in range(500, 700):
        #     u -= 7.0
    
        xs[:, k + 1] = cart.rk4_step(xs[:, k], u)

        # print(f"Step {k+1}/{steps-1}, State: {xs[:, k+1]}, Control: {us[:, k]}")
    
    # Save data run
    # xsave = xs[:, :k+1]
    # usave = us[:, :k+1]
    # np.save("x_data", xsave)
    # np.save("u_data", usave)
    # exit()


    cost = np.trace(xs.T @ cart.Q @ xs) + np.trace(us.T @ cart.R @ us)
    print(f"Total cost over the simulation: {cost}")

    # VISUALIZATION
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    
    # Draw cart
    rec = patches.Rectangle((-0.5, -0.1), 1, 0.2, fc='blue')
    rec.set_animated(True)
    ax.add_patch(rec)
    
    # Draw pendulum
    line, = ax.plot([], [], 'o-', lw=2)

    # Create a title object
    title = ax.text(0.05, 0.95, '', transform=ax.transAxes, ha='left')
    title.set_animated(True)

    def update_plot(frame):
        x = xs[0, frame-1]
        theta = xs[1, frame-1]

        # Update cart position
        rec.set_xy((x - 0.5, -0.1))
        
        # Update pendulum position
        pendulum_x = [x, x - l * np.sin(theta)]
        pendulum_y = [0, l * np.cos(theta)]
        line.set_data(pendulum_x, pendulum_y)

        # Update the title text
        title.set_text(f'Time: {frame * T:.2f} s')

        return rec, line, title
    
    def init():
        rec.set_xy((-0.5, -0.1))
        line.set_data([], [])
        title.set_text('')
        return rec, line, title
    
    ani = animation.FuncAnimation(
        fig,
        func=update_plot,
        frames=range(1, steps, 20),
        interval=1,
        blit=True,
        init_func=init
    )
    plt.show()

    # Save the animation as a GIF file
    ani.save(gif_name, writer='pillow', fps=60)

    # Plot state trajectories and control inputs
    time = np.arange(steps) * T
    plt.figure(figsize=(10, 10))
    plt.subplot(5, 1, 1)
    plt.plot(time, xs[0, :], label='Cart Position (x)')
    plt.ylabel('Position (m)')
    plt.legend()

    plt.subplot(5, 1, 2)
    plt.plot(time, xs[1, :], label='Pendulum Angle (theta)',
                color='orange')
    plt.ylabel('Angle (rad)')
    plt.legend()

    plt.subplot(5, 1, 3)
    plt.plot(time, xs[2, :], label='Cart Velocity (x_dot)',
                color='green')
    plt.ylabel('Velocity (m/s)')
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.plot(time, xs[3, :], label='Pendulum Angular Velocity (theta_dot)',
                color='red')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.subplot(5, 1, 5)
    plt.step(time[:-1], us[0, :], label='Control Input (u)', where='post', color='purple')
    plt.ylabel('Force (N)')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()

