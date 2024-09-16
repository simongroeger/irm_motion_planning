import numpy as np
import time
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt

import jax
import jax.numpy as jnp

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation


def plot_loss_contour(fig, axs):

    # make these smaller to increase the resolution
    dx, dy = 0.1, 0.1

    # generate 2 2d grids for the x & y bounds
    x, y = np.mgrid[slice(-4, 4 + dy, dy),
                    slice(-4, 4 + dx, dx)]

    #z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
    z = compute_point_obstacle_cost(x,y)

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


def compute_point_obstacle_cost(self, x,y):
    cost = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = jnp.array([[x[i,j]], [y[i,j]]])
            cost[i,j] = self.compute_cartesian_cost(point, 0).item()
    return cost

# Creates the animation.
def create_animation(km, dkm, jac, data, losses, aux, u_loss):
    
    fig, ((ax0, ax2, ax4), (ax1, ax5, ax3)) = plt.subplots(nrows=2, ncols=3)
    last_id = max(data.keys())

    # Plot 2d workspace cost potential
    plot_loss_contour(fig, [ax0, ax2])

    #plot start and goal
    start_cart = fk(start_config)
    ax0.plot(start_cart[0], start_cart[1], 'o', color="yellow", label="start_config")
    ax2.plot(start_cart[0], start_cart[1], 'o', color="yellow", label="start_config")

    goal_cart = fk(goal_config)
    ax0.plot(goal_cart[0], goal_cart[1], 'o', color="gold", label="goal_config")
    ax2.plot(goal_cart[0], goal_cart[1], 'o', color="gold", label="goal_config")
 
    cartesian_data = fk(straight_line)
    ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='tab:gray', label="initial straight line trajectory")

    init_trajectory = evaluate(data[0], km, jac)
    final_trajectory = evaluate(data[last_id], km, jac)

    cartesian_data = fk(final_trajectory)
    ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='tab:gray', label="final ee trajectory")

    ax0.plot([0], [0], 'o', color="black", label="joint 0")
    ax2.plot([0], [0], 'o', color="black", label="joint 0")
   
    cartesian_data = fk_joint(init_trajectory, 1)
    curr_fx1, = ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='blue', label="joint 1")
    cartesian_data = fk_joint(init_trajectory, 2)
    curr_fx2, = ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='orange', label="joint 2")
    cartesian_data = fk(init_trajectory)
    curr_fx, = ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='darkgreen', label="ee")

    #big iter
    lists = sorted(aux.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    ax4.plot(x, y, label="big_iter")
    lists = sorted(losses.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    ax4.plot(x, y, label="loss")
    lists = sorted(u_loss.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    ax4.plot(x, y, label="u_loss")
    

    # trajectory point cost over iteration
    trajectory_point_cost = np.zeros(N_timesteps)
    for i in range(N_timesteps):
        trajectory_point_cost[i] = compute_trajectory_obstacle_cost(final_trajectory[i]).item()
    ax3.plot(t, trajectory_point_cost, '-', color='grey')
    for i in range(N_timesteps):
        trajectory_point_cost[i] = compute_trajectory_obstacle_cost(init_trajectory[i]).item()
    ax3.plot(t, trajectory_point_cost, '-', color='grey')
    s5, = ax3.plot(t, trajectory_point_cost, '-', color='black')


    # final robot movement
    fin_movement = np.zeros((4, 2, N_timesteps))
    fin_movement[1] = fk_joint(final_trajectory, 1)
    fin_movement[2] = fk_joint(final_trajectory, 2)
    fin_movement[3] = fk_joint(final_trajectory, 3)
    s4,  = ax2.plot(fin_movement[:,0,0], fin_movement[:,1,0], '-', color = 'black', label="robot")


    # joint position over iteration
    ax1.plot(t, straight_line[:, 0], '-', color='grey')
    ax1.plot(t, straight_line[:, 1], '-', color='grey')
    ax1.plot(t, straight_line[:, 2], '-', color='grey')

    ax1.plot(t, final_trajectory[:, 0], '-', color='grey')
    ax1.plot(t, final_trajectory[:, 1], '-', color='grey')
    ax1.plot(t, final_trajectory[:, 2], '-', color='grey')

    s1, = ax1.plot(t, init_trajectory[:, 0], '-', color='blue', label="joint 0")
    s2, = ax1.plot(t, init_trajectory[:, 1], '-', color='orange', label="joint 1")
    s3, = ax1.plot(t, init_trajectory[:, 2], '-', color='green', label="joint 2")


    # joint velocities over iterations
    init_joint_velocity = evaluate(data[0], dkm, jac)
    ax5.plot(t, init_joint_velocity[:, 0], '-', color='grey')
    ax5.plot(t, init_joint_velocity[:, 1], '-', color='grey')
    ax5.plot(t, init_joint_velocity[:, 2], '-', color='grey')

    final_joint_velocity = evaluate(data[last_id], dkm, jac)
    ax5.plot(t, final_joint_velocity[:, 0], '-', color='grey')
    ax5.plot(t, final_joint_velocity[:, 1], '-', color='grey')
    ax5.plot(t, final_joint_velocity[:, 2], '-', color='grey')

    s6, = ax5.plot(t, final_joint_velocity[:, 0], '-', color='blue', label="joint 0")
    s7, = ax5.plot(t, final_joint_velocity[:, 1], '-', color='orange', label="joint 1")
    s8, = ax5.plot(t, final_joint_velocity[:, 2], '-', color='green', label="joint 2")


    # Title.
    title_text = 'Trajectory Optimization \n Iteration %d, Loss %.2f'
    title = ax3.text(0.5, 0.5, title_text, horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14)

    ax0.legend(loc='lower left')
    ax2.legend(loc='lower left', title="final robot movement")
    ax4.legend(title="opt iter")
    ax1.legend(title="joint position")
    ax5.legend(title="joint velocity")
    ax3.legend(title="trajectory cost")


    # Init only required for blitting to give a clean slate.
    def init():
        return animate(0)
        
    # Update at each iteration.
    def animate(iteration):
        if iteration % 1 == 0:
            title.set_text(title_text % (iteration, losses[iteration]))

            trajectory = evaluate(data[iteration], km, jac)
            cartesian_data = fk(trajectory)
            curr_fx.set_xdata(cartesian_data[0])
            curr_fx.set_ydata(cartesian_data[1])

            cartesian_data = fk_joint(trajectory,1)
            curr_fx1.set_xdata(cartesian_data[0])
            curr_fx1.set_ydata(cartesian_data[1])

            cartesian_data = fk_joint(trajectory,2)
            curr_fx2.set_xdata(cartesian_data[0])
            curr_fx2.set_ydata(cartesian_data[1])

            trajectory_point_cost = np.zeros(N_timesteps)
            for i in range(N_timesteps):
                trajectory_point_cost[i] = compute_trajectory_obstacle_cost(trajectory[i]).item()
            s5.set_ydata(trajectory_point_cost)

            s1.set_ydata(trajectory[:, 0])
            s2.set_ydata(trajectory[:, 1])
            s3.set_ydata(trajectory[:, 2])

            joint_velocity = evaluate(data[iteration], dkm, jac)
            s6.set_ydata(joint_velocity[:, 0])
            s7.set_ydata(joint_velocity[:, 1])
            s8.set_ydata(joint_velocity[:, 2])
        
            
            ri = int(iteration * N_timesteps / max(data.keys()))
            s4.set_xdata(fin_movement[:,0,ri])
            s4.set_ydata(fin_movement[:,1,ri])

        return curr_fx, curr_fx1, curr_fx2, s1, s2, s3, s4, s5, s6, s7, s8, title,

    ani = FuncAnimation(fig, animate, max(data.keys()), init_func=init, interval=200, blit=True, repeat=False)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    #ani.save('to_mg.gif', fps=30)


create_animation(km, dkm, jac, data, losses, aux, u_loss)
