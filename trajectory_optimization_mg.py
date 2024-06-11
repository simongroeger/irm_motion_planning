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
N_timesteps = 100
N_joints = 3

rbf_var = 0.2

max_iteration = 500
lr_start = 0.01
lr_end =   0.0001
lambda_reg = 0.1
lambda_constraint = 2

max_joint_velocity = 3
min_joint_position = -1
max_joint_position = 2


obstacles = jnp.array([ [ 2, -3],
                        [-2, 2],
                        [3, 3],
                        [-1, -2],
                        [-2, 1],
                        [-1, -1],
                        [-2, -3],
                        [-2, 0],
                        [1, 3],
                        [3, 2],
                        [2, 3]])


link_length = jnp.array([1.5, 1.0, 0.5])

start_config = jnp.array([0.0, 0.0, 0.0])
goal_config = jnp.array([1.2, 0.8, 0.3])


straight_line = jnp.stack((
    jnp.linspace(start_config[0], goal_config[0], N_timesteps),
    jnp.linspace(start_config[1], goal_config[1], N_timesteps),
    jnp.linspace(start_config[2], goal_config[2], N_timesteps)    
)).T


def lr(iter):
    return lr_start + (lr_end - lr_start) * iter / max_iteration


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





def jacobian(config):
    c2 = config.reshape(-1, 3)
    c = jnp.cumsum(c2,axis=1)
    
    x = - jnp.mul(link_length, jnp.sin(c))
    reverse_cumsum_x = x + jnp.sum(x,axis=1) - jnp.cumsum(x,axis=1)

    y = jnp.mul(link_length, jnp.cos(c))
    reverse_cumsum_y = y + jnp.sum(y,axis=1) - jnp.cumsum(y,axis=1)
    
    j = jnp.stack((reverse_cumsum_x, reverse_cumsum_y))
    return j



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


def fk(config):
    c2 = config.reshape(-1, 3)
    c = jnp.cumsum(c2,axis=1)
    pos_x = link_length @ jnp.cos(c).T
    pos_y = link_length @ jnp.sin(c).T
    pos = jnp.stack((pos_x, pos_y))
    return pos


def fk_joint(config, joint_id):
    c2 = config.reshape(-1, 3)[:, :joint_id].reshape(-1, joint_id)
    c = jnp.cumsum(c2,axis=1)
    ll = link_length[:joint_id]
    pos_x = ll @ jnp.cos(c).T
    pos_y = ll @ jnp.sin(c).T
    pos = jnp.stack((pos_x, pos_y))
    return pos


def init_trajectory():
    t = jnp.linspace(0, 1, N_timesteps)
    kernel_matrix = create_kernel_matrix(rbf_kernel, t, t)

    #jc = 0.5 * (start_config + goal_config)
    #a = jacobian(jc)[0]
    #unjac = jacobian(jc)[0].T @ jacobian(jc)[0]
    #jac = unjac / jnp.mean(unjac)
    jac = jnp.eye(3) + jax.random.normal(jax.random.PRNGKey(0), (3,3)) / 5

    #alpha = jax.random.normal(jax.random.PRNGKey(0), (N_timesteps, N_joints))
    kmm = jnp.max(jnp.sum(kernel_matrix, axis=0)) * jnp.max(jnp.sum(jac, axis=0))
    alpha = jnp.linspace(start_config/kmm, goal_config/kmm, N_timesteps) + jax.random.normal(jax.random.PRNGKey(0), (N_timesteps, N_joints)) / 100

    lr = 0.01
    lambda_reg = 0.02

    #fit_trajectory_to_straigth_line
    y = straight_line.clone()

    for iteration in range(20):

        # Evaluate fx using the current alpha.
        fx = evaluate(alpha, kernel_matrix, jac)

        # Compute loss (just for logging!).
        loss = jnp.sum(jnp.square(y - fx)) + lambda_reg * jnp.sum((jnp.matmul(alpha.T, fx)))
        print('Init %d: Loss = %0.3f' % (iteration, loss))

        # Compute gradient and update.
        alpha = 2 * lr * (y - fx) + (1 - 2 * lambda_reg * lr) * alpha

    return t, alpha, kernel_matrix, jac


def compute_cartesian_cost(f):
    t_len = f.shape[1]
    o_len = obstacles.shape[0]
    f_expand = jnp.expand_dims(f, 2) @ jnp.ones((1, 1, o_len))
    o_expand = jnp.expand_dims(obstacles, 2) @ jnp.ones((1, 1, t_len))
    o_reshape = o_expand.transpose((1,2,0))

    #cost_v = jnp.sum(0.8 / (0.5 + jnp.sqrt(jnp.sum(jnp.square(f_expand - o_reshape), axis=0))), axis=1)
    cost_v = jnp.sum(0.8 / (0.5 + jnp.linalg.norm(f_expand - o_reshape, axis=0)), axis=1)
    cost = jnp.max(cost_v) + jnp.sum(cost_v) / cost_v.shape[0]
    return cost

def derivative_cartesian_cost(f):
    t_len = f.shape[1]
    o_len = obstacles.shape[0]
    f_expand = jnp.expand_dims(f, 2) @ jnp.ones((1, 1, o_len))
    o_expand = jnp.expand_dims(obstacles, 2) @ jnp.ones((1, 1, t_len))
    o_reshape = o_expand.transpose((1,2,0))

    # cost_v = jnp.sum(0.8 / (0.5 + jnp.sum(jnp.square(f_expand - o_reshape), axis=0)), axis=1)
    # cost = jnp.max(cost_v) + jnp.sum(cost_v) / cost_v.shape[0]
    
    a = jnp.sum(-0.8 / jnp.square(0.5 + jnp.sum(2 * (f_expand - o_reshape), axis=0)), axis=1)

    return 0


def compute_point_obstacle_cost(x,y):
    cost = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = jnp.array([[x[i,j]], [y[i,j]]])
            cost[i,j] = compute_cartesian_cost(point).item() / 2
    return cost


def compute_trajectory_obstacle_cost(trajectory):
    f = fk(trajectory)
    f1 = fk_joint(trajectory, 1)
    f2 = fk_joint(trajectory, 2)
    return (compute_cartesian_cost(f) + compute_cartesian_cost(f1) + compute_cartesian_cost(f2)) / 3


# constrained loss
def start_goal_cost(trajectory):
    s = trajectory[0]
    g = trajectory[N_timesteps-1]
    loss = jnp.linalg.norm(s-start_config) + jnp.linalg.norm(g-goal_config)
    return loss

def joint_limit_cost(trajectory):
    loss = jnp.sum(jnp.exp(100*(trajectory - max_joint_position)) + jnp.exp(100*(-(trajectory - min_joint_position)))) / N_timesteps
    return loss

def joint_velocity_limit_cost(trajectory):
    joint_abs_velocity = jnp.abs(trajectory[1 : ] - trajectory[ : len(trajectory)-1]) * N_timesteps
    loss = jnp.sum(jnp.exp(100*(joint_abs_velocity - max_joint_velocity))) / N_timesteps
    return loss

def compute_trajectory_cost(alpha, km, jac):
    trajectory = evaluate(alpha, km, jac)
    return compute_trajectory_obstacle_cost(trajectory) + lambda_constraint * (start_goal_cost(trajectory)) #+ joint_limit_cost(trajectory) + joint_velocity_limit_cost(trajectory))




t, alpha, km, jac = init_trajectory()

data = {}
losses = {}

data[0] = evaluate(alpha, km, jac)
losses[0] = 0

print(jac)


l = jax.jit(compute_trajectory_cost)
g = jax.jit(jax.grad(compute_trajectory_cost))

st = time.time()

for iter in range(max_iteration):

    loss = l(alpha, km, jac)

    if iter % 10 == 0: 
        print(iter, loss.item())

        data[iter] = evaluate(alpha, km, jac)
        losses[iter] = loss

    # constrained gradient descent
    # alpha.data = (1 - lambda_reg * lr(iter)) * alpha.data - lr(iter) * alpha.grad.data
    

    alpha_grad = g(alpha, km, jac)

    alpha = (1 - lambda_reg * lr(iter)) * alpha - lr(iter) * alpha_grad


et = time.time()
print("took", 1000*(et-st), "ms")


#create_animation(data, losses)



