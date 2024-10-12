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

    parser.add_argument('--animate', type=lambda x: (str(x).lower() == 'true'), default=True)

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

    parser.add_argument('--vis-legend', type=lambda x: (str(x).lower() == 'true'),          default=True)
    parser.add_argument('--vis-sgb', type=lambda x: (str(x).lower() == 'true'),             default=True)
    parser.add_argument('--vis-sg-robot', type=lambda x: (str(x).lower() == 'true'),        default=True)
    parser.add_argument('--vis-obstacles', type=lambda x: (str(x).lower() == 'true'),       default=False)
    parser.add_argument('--vis-straight-line', type=lambda x: (str(x).lower() == 'true'),   default=False)
    parser.add_argument('--vis-gradient', type=lambda x: (str(x).lower() == 'true'),        default=False)
    parser.add_argument('--vis-final-ee', type=lambda x: (str(x).lower() == 'true'),        default=False)
    parser.add_argument('--vis-final-joints', type=lambda x: (str(x).lower() == 'true'),    default=True)
    parser.add_argument('--vis-final-robot', type=lambda x: (str(x).lower() == 'true'),     default=False)


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
        #fig.colorbar(cf, ax=ax)
    
    fig.tight_layout()


def compute_point_obstacle_cost_g(x, y, env):
    cost = np.zeros((x.shape[0], x.shape[1], 2))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = jnp.array([[x[i,j]], [y[i,j]]])
            cost[i,j] = environment.compute_cost_g(point, env.obstacles)[:, 0]
    return cost


def plot_gradient(fig, axs, env):

    # make these smaller to increase the resolution
    dx, dy = 0.5, 0.5

    # generate 2 2d grids for the x & y bounds
    x = slice(-4, 4 + dx, dx)
    y = slice(-4, 4 + dy, dy)
    X, Y = np.mgrid[x, y]

    z = -1 * compute_point_obstacle_cost_g(X,Y, env)

    for ax in axs:
        q = ax.quiver(X, Y, z[:, :, 0], z[:, :, 1], width=0.002)
        #ax.quiverkey(q, X=0.3, Y=1.1, U=10, label='Quiver key, length = 10', labelpos='E')


args = parse_args()
tr = Trajectory(args)
robot = Robot(args)
env = Environment()
#env.start_config = jnp.array([0.0, 0.5, -1.0])

fig, (ax0) = plt.subplots(nrows=1, ncols=1)
fig.dpi = 1000/16*1.5

# Plot 2d workspace cost potential
plot_loss_contour(fig, [ax0], env)

if args.vis_gradient:
    plot_gradient(fig, [ax0], env)

if args.vis_obstacles:
    ax0.scatter(env.obstacles[:, 0], env.obstacles[:, 1], 2000, color="red", linewidths=5, edgecolors="black", label="obstacles")

if args.vis_sgb:
    #plot start and goal
    start_cart = robot.fk(env.start_config)
    ax0.scatter(start_cart[0], start_cart[1], 100, color="yellow", linewidths=0.5, edgecolors="black", label="start_config")

    goal_cart = robot.fk(env.goal_config)
    ax0.scatter(goal_cart[0], goal_cart[1], 100, color="gold", linewidths=0.5, edgecolors="black", label="goal_config")

    #plot robot base
    ax0.scatter([0], [0], 50, color="black", label="joint 1")

if args.vis_sg_robot:
    sg_trajectory = jnp.array([env.start_config, env.goal_config])
    fin_movement = np.zeros((4, 2, 2))
    fin_movement[1] = robot.fk_joint_1(sg_trajectory)
    fin_movement[2] = robot.fk_joint_2(sg_trajectory)
    fin_movement[3] = robot.fk_joint_3(sg_trajectory)
    ax0.plot(fin_movement[:,0, 0], fin_movement[:,1, 0], linewidth=3, c='black')
    ax0.plot(fin_movement[:,0, 1], fin_movement[:,1, 1], linewidth=3, c='black')


trajectory_series = np.loadtxt("trajectory_series.txt").reshape((-1, args.n_timesteps, args.n_joints))
print(trajectory_series.shape)
trajectory = trajectory_series[0]

#trajectory_labels=["0.0", "0.25", "0.5", "0.75", "1.0"]
#for label in trajectory_labels:
#    trajectories.append(np.loadtxt("trajectory_result_" + label + ".txt"))

N_timesteps = len(trajectory)
    
t = jnp.linspace(0, 1, N_timesteps)
c = 6 * t**5 - 15 * t**4 + 10 * t**3
straight_line = env.start_config + (env.goal_config - env.start_config) * c[:, jnp.newaxis]
cartesian_data = robot.fk(straight_line)
if args.vis_straight_line:
    ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='tab:gray', label="initial straight line trajectory")


# robot movement
fin_movement = np.zeros((4, 2, N_timesteps))
fin_movement[1] = robot.fk_joint_1(trajectory)
fin_movement[2] = robot.fk_joint_2(trajectory)
fin_movement[3] = robot.fk_joint_3(trajectory)

if args.vis_final_ee:
    fe, = ax0.plot(fin_movement[3,0], fin_movement[3,1], '-', c='black', label="final ee trajectory")


if args.vis_final_joints:
        fj1, = ax0.plot(fin_movement[1,0], fin_movement[1,1], '-', c="blue", linewidth=2, label="joint 2")
        fj2, = ax0.plot(fin_movement[2,0], fin_movement[2,1], '-', c="orange", linewidth=2, label="joint 3")
        fj3, = ax0.plot(fin_movement[3,0], fin_movement[3,1], '-', c="darkgreen", linewidth=2, label="end effector")


if args.vis_final_robot:
    for i in range(N_timesteps):
        alpha = 1 if i in [0, N_timesteps-1] else 0.5
        for j, color in enumerate(["blue", "orange", "darkgreen"]):
            ax0.plot(fin_movement[j:j+2,0,i], fin_movement[j:j+2,1,i], color=color, linewidth=2, alpha=alpha)
            #ax0.plot(fin_movement[:,0,i], fin_movement[:,1,i], 'o', color = 'tab:grey', alpha=alpha)

if args.vis_legend:
    ax0.legend(loc='lower left')

plt.subplots_adjust(wspace=0.1, hspace=0.1)

if args.animate:
    def update(frame):
        trajectory = trajectory_series[frame]

        fin_movement = np.zeros((4, 2, N_timesteps))
        fin_movement[1] = robot.fk_joint_1(trajectory)
        fin_movement[2] = robot.fk_joint_2(trajectory)
        fin_movement[3] = robot.fk_joint_3(trajectory)
        
        if args.vis_final_ee:
            fe.set_xdata(fin_movement[3,0])
            fe.set_ydata(fin_movement[3,1])

        if args.vis_final_joints:
            fj1.set_xdata(fin_movement[1,0])
            fj1.set_ydata(fin_movement[1,1])
            fj2.set_xdata(fin_movement[2,0])
            fj2.set_ydata(fin_movement[2,1])
            fj3.set_xdata(fin_movement[3,0])
            fj3.set_ydata(fin_movement[3,1])

    ani = FuncAnimation(fig=fig, func=update, frames=len(trajectory_series), interval=250)
    ani.save('trajectory_series.gif', writer='imagemagick')

plt.show()


