import numpy as np
import time
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt

import jax
import jax.numpy as jnp

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation


import os
os.environ['TF_XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

#param
N_timesteps = 20
N_joints = 3

rbf_var = 0.2

max_iteration = 3
lr_start = 0.001
lr_end =   0.001
lambda_reg = 0.4
lambda_constraint = 2

track = jnp.array(np.loadtxt("motion_planning.txt")[:,:2])

def lr(iter):
    return lr_start + (lr_end - lr_start) * iter / max_iteration

"""
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



# Creates the animation.
def create_animation(data, losses):
    
    fig, ((ax0, ax2, ax4), (ax1, ax5, ax3)) = plt.subplots(nrows=2, ncols=3)
    last_id = max(data.keys())
    t = jnp.linspace(0, 1, N_timesteps)

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

    ax0.plot([0], [0], 'o', color="black", label="joint 0")
    ax2.plot([0], [0], 'o', color="black", label="joint 0")
   
    cartesian_data = fk_joint(data[0], 1)
    curr_fx1, = ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='blue', label="joint 1")
    cartesian_data = fk_joint(data[0], 2)
    curr_fx2, = ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='orange', label="joint 2")
    cartesian_data = fk(data[0])
    curr_fx, = ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='darkgreen', label="ee")
    

    # trajectory point cost over iteration
    trajectory_point_cost = np.zeros(N_timesteps)
    for i in range(N_timesteps):
        trajectory_point_cost[i] = compute_trajectory_obstacle_cost(data[last_id][i]).item() / 2
    ax3.plot(t, trajectory_point_cost, '-', color='grey')
    for i in range(N_timesteps):
        trajectory_point_cost[i] = compute_trajectory_obstacle_cost(data[0][i]).item() / 2
    ax3.plot(t, trajectory_point_cost, '-', color='grey')
    s5, = ax3.plot(t, trajectory_point_cost, '-', color='black')


    # final robot movement
    fin_movement = np.zeros((4, 2, N_timesteps))
    fin_movement[1] = fk_joint(data[last_id], 1)
    fin_movement[2] = fk_joint(data[last_id], 2)
    fin_movement[3] = fk_joint(data[last_id], 3)
    s4,  = ax2.plot(fin_movement[:,0,0], fin_movement[:,1,0], '-', color = 'black', label="robot")


    # joint position over iteration
    ax1.plot(t, straight_line[:, 0], '-', color='grey')
    ax1.plot(t, straight_line[:, 1], '-', color='grey')
    ax1.plot(t, straight_line[:, 2], '-', color='grey')

    ax1.plot(t, data[last_id][:, 0], '-', color='grey')
    ax1.plot(t, data[last_id][:, 1], '-', color='grey')
    ax1.plot(t, data[last_id][:, 2], '-', color='grey')

    s1, = ax1.plot(t, data[0][:, 0], '-', color='blue', label="joint 0")
    s2, = ax1.plot(t, data[0][:, 1], '-', color='orange', label="joint 1")
    s3, = ax1.plot(t, data[0][:, 2], '-', color='green', label="joint 2")


    # joint velocities over iterations
    joint_velocity = (data[last_id][1 : ] - data[last_id][ : len(data[last_id])-1]) * N_timesteps
    s6, = ax5.plot(t[:N_timesteps-1], joint_velocity[:, 0], '-', color='blue', label="joint 0")
    s7, = ax5.plot(t[:N_timesteps-1], joint_velocity[:, 1], '-', color='orange', label="joint 1")
    s8, = ax5.plot(t[:N_timesteps-1], joint_velocity[:, 2], '-', color='green', label="joint 2")


    # Title.
    title_text = 'Trajectory Optimization \n Iteration %d, Loss %.2f'
    title = ax0.text(0.8, 0.9, title_text, horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14)

    ax0.legend(loc='lower left')
    ax2.legend(loc='lower left', title="final robot movement")
    ax1.legend(title="joint position")
    ax5.legend(title="joint velocity")
    ax3.legend(title="trajectory cost")


    # Init only required for blitting to give a clean slate.
    def init():
        return animate(0)
        
    # Update at each iteration.
    def animate(iteration):
        if iteration % 10 == 0:
            title.set_text(title_text % (iteration, losses[iteration]))

            cartesian_data = fk(data[iteration])
            curr_fx.set_xdata(cartesian_data[0])
            curr_fx.set_ydata(cartesian_data[1])

            cartesian_data = fk_joint(data[iteration],1)
            curr_fx1.set_xdata(cartesian_data[0])
            curr_fx1.set_ydata(cartesian_data[1])

            cartesian_data = fk_joint(data[iteration],2)
            curr_fx2.set_xdata(cartesian_data[0])
            curr_fx2.set_ydata(cartesian_data[1])

            trajectory_point_cost = np.zeros(N_timesteps)
            for i in range(N_timesteps):
                trajectory_point_cost[i] = compute_trajectory_obstacle_cost(data[iteration][i]).item() / 2
            s5.set_ydata(trajectory_point_cost)


            s1.set_ydata(data[iteration][:, 0])
            s2.set_ydata(data[iteration][:, 1])
            s3.set_ydata(data[iteration][:, 2])

            joint_velocity = (data[iteration][1 : ] - data[iteration][ : len(data[iteration])-1]) * N_timesteps
            s6.set_ydata(joint_velocity[:, 0])
            s7.set_ydata(joint_velocity[:, 1])
            s8.set_ydata(joint_velocity[:, 2])
        
            
            ri = int(iteration * N_timesteps / max(data.keys()))
            s4.set_xdata(fin_movement[:,0,ri])
            s4.set_ydata(fin_movement[:,1,ri])

        return curr_fx, curr_fx1, curr_fx2, s1, s2, s3, s4, s5, s6, s7, s8, title,

    ani = FuncAnimation(fig, animate, max(data.keys()), init_func=init, interval=20, blit=True, repeat=False)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    #ani.save('to_mg.gif', fps=30)

"""



def rbf_kernel(x_1, x_2):
    return jnp.exp( - (x_1 - x_2)**2 / (2 * rbf_var**2) )


# Create kernel matrix from dataset.
def create_kernel_matrix(kernel_f, x, x2):
    a, b = jnp.meshgrid(x, x2)
    kernel_matrix = kernel_f(a,b)
    return kernel_matrix

def evaluate(alpha, kernel_matrix, jac):
    return kernel_matrix @ alpha @ jac

def eval_any(alpha, kernel_f, support_x, eval_x, jac):
    return evaluate(alpha, create_kernel_matrix(kernel_f, eval_x, support_x), jac)



def init_trajectory():

    sum = 0
    t = [0]
    for i in range(1, len(track)):
        dist = jnp.linalg.norm(track[i] - track[i-1])
        sum += dist
        t.append(sum)
    
    t = jnp.array(t)
    t /= sum

    kernel_matrix = create_kernel_matrix(rbf_kernel, t, t)
    jac = jnp.eye(2) + jax.random.normal(jax.random.PRNGKey(0), (2,2)) / 5

    #fit_trajectory_to_straigth_line 
    alpha = jnp.linalg.solve(kernel_matrix, track @ jnp.linalg.inv(jac))

    lambda_reg = 0.02
    fx = evaluate(alpha, kernel_matrix, jac)
    loss = jnp.sum(jnp.linalg.norm(track - fx)) #+ lambda_reg * jnp.sum((jnp.matmul(alpha.T, fx)))
    print('Init loss = %0.3f' % ( loss))

    return t, alpha, kernel_matrix, jac


def compute_trajectory_cost(alpha, km, jac):
    trajectory = evaluate(alpha, km, jac)
    loss = jnp.sum(jnp.exp(10*(jnp.abs(trajectory - track)- 1 )))
    return loss




def main():

    t, alpha, km, jac = init_trajectory()

    data = {}
    losses = {}

    data[0] = evaluate(alpha, km, jac)
    losses[0] = 0

    l = jax.jit(compute_trajectory_cost)
    g = jax.jit(jax.grad(compute_trajectory_cost))


    for iter in range(0):

        loss = l(alpha, km, jac)

        if iter % 1 == 0: 
            print(iter, loss)

            data[iter] = evaluate(alpha, km, jac)
            losses[iter] = loss

        alpha_grad = g(alpha, km, jac)

        alpha = (1 - lambda_reg * lr(iter)) * alpha - lr(iter) * alpha_grad


    return data, losses, t, alpha, km, jac


st = time.time()

m = jax.jit(main)
data, losses, t, alpha, km, jac = main()

et = time.time()
print("took", 1000*(et-st), "ms")

#create_animation(data, losses)

trajectory = evaluate(alpha, km, jac)

plt.plot(trajectory[:, 1], trajectory[:, 0], label="trajectory")
plt.plot(track[:, 1], track[:, 0], label="track")
plt.legend()
plt.show()





