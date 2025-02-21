import numpy as np
import time
import argparse

import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt

import jax
import jax.numpy as jnp

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation

import sys
sys.path.insert(0, '..')

np.set_printoptions(precision=4)

import environment
from environment import Environment
from robot import Robot
from trajectory import Trajectory

def parse_args():
    parser = argparse.ArgumentParser()

    # trajectory
    parser.add_argument('--n-timesteps', type=float, default=50)
    parser.add_argument('--rbf-variance', type=float, default=0.1)
    parser.add_argument('--jac-gaussian-mean', type=float, default=0.2)
    parser.add_argument('--constraint-violating-dependant-loss', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--joint-safety-limit', type=float, default=0.98)

    # robot
    parser.add_argument('--n-joints', type=int, default=3)
    parser.add_argument('--link-length', type=float, nargs='+', default=[1.5, 1.0, 0.5])
    parser.add_argument('--max-joint-velocity', type=float, default=5)
    parser.add_argument('--max-joint-position', type=float, default=2)
    parser.add_argument('--min-joint-position', type=float, default=-1)
    
    parser.add_argument('--eps-position', type=float, default=0.01)
    parser.add_argument('--eps-velocity', type=float, default=0.01)

    args = parser.parse_args()
    return args


def compute_point_obstacle_cost(x, y, env):
    cost = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = jnp.array([[x[i,j]], [y[i,j]]])
            cost[i,j] = environment.compute_cost(point, env.obstacles).item()
    return cost


def plot_loss_contour(fig, axs, env):

    # make these smaller to increase the resolution
    dx, dy = 0.1, 0.1

    # generate 2 2d grids for the x & y bounds
    x, y = np.mgrid[slice(-4, 4 + dy, dy),
                    slice(-4, 4 + dx, dx)]

    #z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
    z = compute_point_obstacle_cost(x,y, env)

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

args = parse_args()
tr = Trajectory(args)
robot = Robot(args)
env = Environment()
#env.start_config = jnp.array([0.0, 0.5, -1.0])

trajectory = np.loadtxt("trajectory_result.txt")
N_timesteps = len(trajectory)
    
fig, ((ax0, ax2, ax4), (ax1, ax5, ax3)) = plt.subplots(nrows=2, ncols=3)

# Plot 2d workspace cost potential
plot_loss_contour(fig, [ax0, ax2], env)

#plot start and goal
start_cart = robot.fk(env.start_config)
ax0.plot(start_cart[0], start_cart[1], 'o', color="yellow", label="start_config")
ax2.plot(start_cart[0], start_cart[1], 'o', color="yellow", label="start_config")

goal_cart = robot.fk(env.goal_config)
ax0.plot(goal_cart[0], goal_cart[1], 'o', color="gold", label="goal_config")
ax2.plot(goal_cart[0], goal_cart[1], 'o', color="gold", label="goal_config")

t = jnp.linspace(0, 1, N_timesteps)
c = 6 * t**5 - 15 * t**4 + 10 * t**3
straight_line = env.start_config + (env.goal_config - env.start_config) * c[:, jnp.newaxis]


cartesian_data = robot.fk(straight_line)
ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='tab:gray', label="initial straight line trajectory")

cartesian_data = robot.fk(trajectory)
ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='green', label="final ee trajectory")

ax0.plot([0], [0], 'o', color="black", label="joint 0")
ax2.plot([0], [0], 'o', color="black", label="joint 0")


# trajectory point cost over iteration
trajectory_point_cost = np.zeros(N_timesteps)
for i in range(N_timesteps):
    trajectory_point_cost[i] = tr.compute_trajectory_obstacle_cost(trajectory[i], env.obstacles, 0).item()
ax3.plot(t, trajectory_point_cost, '-', color='grey')


# final robot movement
fin_movement = np.zeros((4, 2, N_timesteps))
fin_movement[1] = robot.fk_joint_1(trajectory)
fin_movement[2] = robot.fk_joint_2(trajectory)
fin_movement[3] = robot.fk_joint_3(trajectory)
ax2.plot(fin_movement[3,0], fin_movement[3,1], '-', c='black', label="final ee trajectory")

for i in range(N_timesteps):
    alpha = 1 if i in [0, N_timesteps-1] else 0.5
    for j, color in enumerate(["blue", "orange", "darkgreen"]):
        ax2.plot(fin_movement[j:j+2,0,i], fin_movement[j:j+2,1,i], color=color, linewidth=2, alpha=alpha)
        #ax2.plot(fin_movement[:,0,i], fin_movement[:,1,i], 'o', color = 'tab:grey', alpha=alpha)


# joint position
ax1.plot(t, straight_line[:, 0], '-', color='grey')
ax1.plot(t, straight_line[:, 1], '-', color='grey')
ax1.plot(t, straight_line[:, 2], '-', color='grey')

ax1.plot(t, trajectory[:, 0], '-', color='blue', label="joint 0")
ax1.plot(t, trajectory[:, 1], '-', color='orange', label="joint 1")
ax1.plot(t, trajectory[:, 2], '-', color='darkgreen', label="joint 2")

# joint velocities 
ax5.set_yticks(np.arange(-10, 10+1, 1.0))
fd_joint_velocity = (trajectory[1:] - trajectory[:-1]) * N_timesteps
ax5.plot(t[:-1], fd_joint_velocity[:, 0], '-', color='blue', label="joint 0")
ax5.plot(t[:-1], fd_joint_velocity[:, 1], '-', color='orange', label="joint 1")
ax5.plot(t[:-1], fd_joint_velocity[:, 2], '-', color='darkgreen', label="joint 2")

ax0.legend(loc='lower left')
ax2.legend(loc='lower left')
ax4.legend(title="opt iter")
ax1.legend(title="joint position", loc='upper left')
ax5.legend(title="joint velocity", loc='upper left')
ax3.legend(title="trajectory cost")

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()


