import numpy as np
import time
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt

import jax
import jax.numpy as jnp

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation


from environment import Environment
from trajectory import Trajectory


def compute_point_obstacle_cost(env, x,y):
    cost = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = jnp.array([[x[i,j]], [y[i,j]]])
            cost[i,j] = env.compute_cost(point).item()
    return cost


def plot_loss_contour(fig, axs, env):

    # make these smaller to increase the resolution
    dx, dy = 0.1, 0.1

    # generate 2 2d grids for the x & y bounds
    x, y = np.mgrid[slice(-4, 4 + dy, dy),
                    slice(-4, 4 + dx, dx)]

    #z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
    z = compute_point_obstacle_cost(env, x,y)

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]

    #draw
    levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

    for ax in axs:
        cf = ax.contourf(x[:-1, :-1] + dx/2.,
                    y[:-1, :-1] + dy/2., z, levels=levels,
                    cmap=plt.get_cmap('PiYG'))
        fig.colorbar(cf, ax=ax)
    
    fig.tight_layout()


env = Environment()
tr = Trajectory()

trajectory = np.loadtxt("gd_trajectory_result.txt")
N_timesteps = len(trajectory)
    
fig, ((ax0, ax2, ax4), (ax1, ax5, ax3)) = plt.subplots(nrows=2, ncols=3)

# Plot 2d workspace cost potential
plot_loss_contour(fig, [ax0, ax2], env)

#plot start and goal
start_cart = env.fk(env.start_config)
ax0.plot(start_cart[0], start_cart[1], 'o', color="yellow", label="start_config")
ax2.plot(start_cart[0], start_cart[1], 'o', color="yellow", label="start_config")

goal_cart = env.fk(env.goal_config)
ax0.plot(goal_cart[0], goal_cart[1], 'o', color="gold", label="goal_config")
ax2.plot(goal_cart[0], goal_cart[1], 'o', color="gold", label="goal_config")

t = jnp.linspace(0, 1, N_timesteps)
c = 3 * t**2 - 2 * t**3
straight_line = jnp.stack((
    env.start_config[0] + (env.goal_config[0] - env.start_config[0]) * c,
    env.start_config[1] + (env.goal_config[1] - env.start_config[1]) * c,
    env.start_config[2] + (env.goal_config[2] - env.start_config[2]) * c
)).T

cartesian_data = env.fk(straight_line)
ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='tab:gray', label="initial straight line trajectory")

cartesian_data = env.fk(trajectory)
ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='green', label="final ee trajectory")

ax0.plot([0], [0], 'o', color="black", label="joint 0")
ax2.plot([0], [0], 'o', color="black", label="joint 0")


# trajectory point cost over iteration
trajectory_point_cost = np.zeros(N_timesteps)
for i in range(N_timesteps):
    trajectory_point_cost[i] = tr.compute_trajectory_obstacle_cost(trajectory[i], 0).item()
ax3.plot(t, trajectory_point_cost, '-', color='grey')


# final robot movement
fin_movement = np.zeros((4, 2, N_timesteps))
fin_movement[1] = env.fk_joint_1(trajectory)
fin_movement[2] = env.fk_joint_2(trajectory)
fin_movement[3] = env.fk_joint_3(trajectory)
ax2.plot(fin_movement[:,0,0], fin_movement[:,1,0], '-', color = 'black', label="robot")


# joint position
ax1.plot(t, straight_line[:, 0], '-', color='grey')
ax1.plot(t, straight_line[:, 1], '-', color='grey')
ax1.plot(t, straight_line[:, 2], '-', color='grey')

ax1.plot(t, trajectory[:, 0], '-', color='grey')
ax1.plot(t, trajectory[:, 1], '-', color='grey')
ax1.plot(t, trajectory[:, 2], '-', color='grey')

# joint velocities 
fd_joint_velocity = (trajectory[1:] - trajectory[:-1]) * N_timesteps
ax5.plot(t[:-1], fd_joint_velocity[:, 0], '-', color='grey')
ax5.plot(t[:-1], fd_joint_velocity[:, 1], '-', color='grey')
ax5.plot(t[:-1], fd_joint_velocity[:, 2], '-', color='grey')

ax0.legend(loc='lower left')
ax2.legend(loc='lower left', title="final robot movement")
ax4.legend(title="opt iter")
ax1.legend(title="joint position")
ax5.legend(title="joint velocity")
ax3.legend(title="trajectory cost")

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()


