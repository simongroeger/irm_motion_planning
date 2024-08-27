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

np.set_printoptions(precision=4)


#param
N_timesteps = 50
N_joints = 3

rbf_var = 0.1

useBLS = True

max_iteration = 100

lr_start = 0.0001
lr_end =   0.0000001

#bls
bls_lr_start = 0.2
bls_alpha = 0.01
bls_beta_minus = 0.5
bls_beta_plus = 1.2
bls_max_iter = 20

lambda_constraint_increase = 10

loop_loss_reduction = 0.001

lambda_reg = 0.0001
lambda_constraint = 0.5
lambda_2_constraint = 0.1
lambda_max_cost = 0.8

trajectory_duration = 2
max_joint_velocity = 3
min_joint_position = -1
max_joint_position = 2
mean_joint_position = 0.5*(max_joint_position + min_joint_position)
std_joint_position = max_joint_position - mean_joint_position

link_length = jnp.array([1.5, 1.0, 0.5])

start_config = jnp.array([0.0, 0.0, 0.0])
goal_config = jnp.array([1.2, 1.0, 0.3])
eps_start_goal_velocity = 0.05
eps_start_goal_distance = 0.01

obs_1 = jnp.array([     
                    [ 2, -3],
                    [-2, 2],
                    [3, 3],
                    [-1, -2],
                    [-2, 1],
                    [-1, -1],
                    [-2, -3],
                    [-2, 0],
                    [1, 3],
                    [3, 2],
                    [2, 3]
                ])


obs_2 = jnp.array([     
                    [3, 3],
                    [3, 2],
                    [3, 1],
                    [2, 3],
                    [2.2, 2.2],
                    [2, 1],
                    [1, 3],
                    [1, 2],
                    [1, 1],
                    ])

obstacles = obs_1

c = 3*jnp.linspace(0, 1, N_timesteps)**2 - 2* jnp.linspace(0, 1, N_timesteps) ** 3
straight_line = jnp.stack((
    start_config[0] + (goal_config[0] - start_config[0]) * c,
    start_config[1] + (goal_config[1] - start_config[1]) * c,
    start_config[2] + (goal_config[2] - start_config[2]) * c
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





def jacobian(config):
    c2 = config.reshape(-1, 3)
    c = jnp.cumsum(c2,axis=1)
    
    x = - jnp.multiply(link_length, jnp.sin(c))
    reverse_cumsum_x = x + jnp.sum(x,axis=1) - jnp.cumsum(x,axis=1)

    y = jnp.multiply(link_length, jnp.cos(c))
    reverse_cumsum_y = y + jnp.sum(y,axis=1) - jnp.cumsum(y,axis=1)
    
    j = jnp.stack((reverse_cumsum_x, reverse_cumsum_y))
    return j



def rbf_kernel(x_1, x_2):
    return jnp.exp( - (x_1 - x_2)**2 / (2 * rbf_var**2) )


def d_rbf_kernel(x_1, x_2):
    return (x_1 - x_2) / (trajectory_duration * rbf_var**2) * jnp.exp( - (x_1 - x_2)**2 / (2 * rbf_var**2) ) 


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
    dkm = create_kernel_matrix(d_rbf_kernel, t, t)
    jac = jnp.eye(3) + jax.random.normal(jax.random.PRNGKey(0), (3,3)) / 5

    #fit_trajectory_to_straigth_line 
    alpha = np.linalg.solve(kernel_matrix, straight_line @ np.linalg.inv(jac))
    lambda_reg = 0.01
    fx = evaluate(alpha, kernel_matrix, jac)
    loss = jnp.sum(jnp.square(straight_line - fx)) + lambda_reg * jnp.sum((jnp.matmul(alpha.T, fx)))
    print('Alpha solve loss = %0.3f' % ( loss))

    return t, alpha, kernel_matrix, jac, dkm


def compute_cartesian_cost(f):
    t_len = f.shape[1]
    o_len = obstacles.shape[0]
    f_expand = jnp.expand_dims(f, 2) @ jnp.ones((1, 1, o_len))
    o_expand = jnp.expand_dims(obstacles, 2) @ jnp.ones((1, 1, t_len))
    o_reshape = o_expand.transpose((1,2,0))

    cost_v = jnp.sum(0.8 / (0.5 + jnp.linalg.norm(f_expand - o_reshape, axis=0)), axis=1)
    max_cost = jnp.max(cost_v)
    avg_cost = jnp.sum(cost_v) / cost_v.shape[0]
    cost = lambda_max_cost * max_cost + (1 - lambda_max_cost) * avg_cost
    return cost


def compute_point_obstacle_cost(x,y):
    cost = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = jnp.array([[x[i,j]], [y[i,j]]])
            cost[i,j] = compute_cartesian_cost(point).item() / (lambda_max_cost + 1)
    return cost


def compute_trajectory_obstacle_cost(trajectory):
    f = fk(trajectory)
    f1 = fk_joint(trajectory, 1)
    f2 = fk_joint(trajectory, 2)
    #return compute_cartesian_cost(f)
    return (compute_cartesian_cost(f) + compute_cartesian_cost(f1) + compute_cartesian_cost(f2)) / 3


def start_goal_position_constraint_fulfilled(trajectory):
    s = trajectory[0]
    if jnp.sum(jnp.square(s-start_config)) > jnp.square(eps_start_goal_distance):
        return False
    g = trajectory[N_timesteps-1]
    if jnp.sum(jnp.square(g-goal_config)) > jnp.square(eps_start_goal_distance):
        return False
    return True

def start_goal_velocity_constraint_fulfilled(joint_velocity):
    if jnp.sum(jnp.square(joint_velocity[0])) > jnp.square(eps_start_goal_velocity):
        return False
    if jnp.sum(jnp.square(joint_velocity[-1])) > jnp.square(eps_start_goal_velocity):
        return False
    return True

def joint_position_constraint(trajectory):
    if trajectory.max() > max_joint_position:
        return False 
    if trajectory.min() < min_joint_position:
        return False 
    return True

def joint_velocity_constraint(joint_velocity):
    if jnp.abs(joint_velocity).max() > max_joint_velocity:
        return False
    return True

def constraintsFulfilled(alpha, km, dkm, jac, verbose=False):
    trajectory = evaluate(alpha, km, jac)
    joint_velocity = evaluate(alpha, dkm, jac)
    
    # start and goal position
    if not start_goal_position_constraint_fulfilled(trajectory):
        if verbose:
            print("violated start goal position", jnp.linalg.norm(trajectory[0]-start_config), jnp.linalg.norm(trajectory[-1]-goal_config))
        return False
    elif verbose:
        print("ok start goal position", jnp.linalg.norm(trajectory[0]-start_config), jnp.linalg.norm(trajectory[-1]-goal_config))

    
    # start and goal velocity
    if not start_goal_velocity_constraint_fulfilled(joint_velocity):
        if verbose:
            print("violated start goal velocity", jnp.linalg.norm(joint_velocity[0]), jnp.linalg.norm(joint_velocity[-1]))
        return False
    elif verbose:
        print("ok start goal velocity", jnp.linalg.norm(joint_velocity[0]), jnp.linalg.norm(joint_velocity[-1]))

    # joint poisiton limit
    if not joint_position_constraint(trajectory):
        if verbose:
            print("joint limit")
        return False
    
    #joint velocity limit
    if not joint_velocity_constraint(joint_velocity):
        if verbose:
            print("joint velocity")
        return False

    return True


# constrained loss
def start_goal_cost(trajectory):
    s = trajectory[0]
    g = trajectory[N_timesteps-1]
    loss = jnp.sum(jnp.square(s-start_config)) + jnp.sum(jnp.square(g-goal_config))
    return loss

def start_goal_velocity_cost(joint_velocity):
    loss = jnp.sum(jnp.square(joint_velocity[0])) + jnp.sum(jnp.square(joint_velocity[-1]))
    return loss

def joint_limit_cost(trajectory):
    loss = jnp.sum(jnp.square((trajectory - mean_joint_position) / std_joint_position)) / N_timesteps
    return loss

def joint_velocity_limit_cost(joint_velocity):
    loss = jnp.sum(jnp.square(joint_velocity / max_joint_velocity)) / N_timesteps
    return loss

def compute_trajectory_cost(alpha, km, dkm, jac):
    trajectory = evaluate(alpha, km, jac)
    joint_velocity = evaluate(alpha, dkm, jac)

    toc = compute_trajectory_obstacle_cost(trajectory) 
    sgpc = start_goal_cost(trajectory)
    sgvc = start_goal_velocity_cost(joint_velocity)
    jpc = joint_limit_cost(trajectory) if not joint_position_constraint(trajectory) else 0
    jvc = joint_velocity_limit_cost(joint_velocity) if not joint_velocity_constraint(joint_velocity) else 0
    return toc + lambda_constraint * (sgpc + sgvc) + lambda_2_constraint * (jpc + jvc)




t, alpha, km, jac, dkm = init_trajectory()

data = {}
losses = {}
u_loss = {}
aux = {}

trajectory = evaluate(alpha, km, jac)

data[0] = alpha
losses[0] = 0
aux[0] = 0
u_loss[0] = 0

last_loss = 1000

l = jax.jit(compute_trajectory_cost)
g = jax.jit(jax.grad(l))

#TODO
l = compute_trajectory_cost
g = jax.grad(l)


st = time.time()

amount_epoch_plot = 1

if not useBLS:
    last_loss = 1000
    for iter in range(max_iteration):

        loss = l(alpha, km, jac)

        if iter % amount_epoch_plot == 0: 
            print(iter, loss.item())

            data[iter] = alpha
            losses[iter] = loss
            aux[iter] = 0
            u_loss[iter] = compute_trajectory_obstacle_cost(data[iter])

            if abs(last_loss - loss) < 0.002 * amount_epoch_plot:
                break

            last_loss = loss
        
        alpha_grad = g(alpha, km, jac)

        alpha = (1 - lambda_reg * lr(iter)) * alpha - lr(iter) * alpha_grad

else:
    outer_loop_iter = 0
    last_outer_loop_increase = -1
    bls_lr = bls_lr_start
    for iter in range(max_iteration):

        loss = l(alpha, km, dkm, jac)

        if iter % amount_epoch_plot == 0: 
            #print(iter, loss.item())
            data[iter] = alpha
            losses[iter] = loss
            aux[iter] = outer_loop_iter
            u_loss[iter] = compute_trajectory_obstacle_cost(evaluate(alpha, km, jac))

        alpha_grad = g(alpha, km, dkm, jac)
        n_alpha_grad = alpha_grad / jnp.linalg.norm(alpha_grad) # normalized

        print("---")
        cf = constraintsFulfilled(alpha, km, dkm, jac, True)
        ul = compute_trajectory_obstacle_cost(evaluate(alpha, km, jac))
        print("iter", iter, "loss", loss, "uloss", ul, "cf", cf)
        for j in range(bls_max_iter):
            new_alpha = (1 - lambda_reg * bls_lr) * alpha - bls_lr * n_alpha_grad
            new_loss = l(new_alpha, km, dkm, jac)
            required_loss = loss - bls_alpha * bls_lr * jnp.sum(alpha_grad * n_alpha_grad)
            cf = constraintsFulfilled(alpha, km, dkm, jac)
            print(" bls_iter", j, "bls_lr", bls_lr, "loss", new_loss, "req loss", required_loss, "constraint", cf)
            
            if new_loss > required_loss:
                bls_lr *= bls_beta_minus
            else:
                print("chose", bls_lr, "with", new_loss)
                alpha = (1 - lambda_reg * bls_lr) * alpha - bls_lr * n_alpha_grad
                bls_lr = bls_lr * bls_beta_plus
                break

        if constraintsFulfilled(alpha, km, dkm, jac, verbose=True):
            if outer_loop_iter > 0 or abs(loss - new_loss) < loop_loss_reduction:
                print("constraints fulfilled and inner loop minimized or one inner loop finished")
                break

        # end of current inner minimzation
        if abs(loss - new_loss) < loop_loss_reduction:
            print("abort too small loss change")
            if iter > last_outer_loop_increase + 1: 
                outer_loop_iter += 1
                last_outer_loop_increase = iter
                print()
                print("NEW BIG ITER; INCREASE LAMBDA", outer_loop_iter)
                lambda_constraint *= lambda_constraint_increase
                lambda_2_constraint *= lambda_constraint_increase
                bls_lr = bls_lr_start
                continue
            else: 
                print("no gradient step possible in big_iter, abort")
                break


et = time.time()
print("took", 1000*(et-st), "ms")


np_trajectory = np.array(evaluate(alpha, km, jac))
np.savetxt("trajectory_result.txt", np_trajectory)
create_animation(km, dkm, jac, data, losses, aux, u_loss)



