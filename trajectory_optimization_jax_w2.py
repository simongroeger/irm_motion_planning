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

    s = trajectory[0]
    g = trajectory[N_timesteps-1]
    #vs = joint_velocity[0]
    #vg = joint_velocity[N_timesteps-1]

    return  jnp.sum(jnp.square(g-goal_config)) < jnp.square(eps_start_goal_distance) #and jnp.sum(jnp.square(vs)) > jnp.square(eps_start_goal_velocity) and jnp.sum(jnp.square(vg)) > jnp.square(eps_start_goal_velocity)

    #jnp.sum(jnp.square(s-start_config)) < jnp.square(eps_start_goal_distance) and
    
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
    return toc + lambda_constraint * (sgpc + sgvc)

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

    init_val = (0, 1000, alpha, bls_lr_start, outer_loop_iter, lambda_constraint)

    
    def t_func(a):
        iter, loss, new_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint = a
        print("constraints fulfilled and inner loop minimized or one inner loop finished")
        return (max_iteration, loss, new_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint)
        
    def ftt_func(a):
        iter, loss, new_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint = a
        outer_loop_iter += 1
        print()
        print("new outer loop", outer_loop_iter, "increase lambda")
        lambda_constraint *= lambda_constraint_increase
        return (iter, loss, new_loss, alpha, bls_lr_start, outer_loop_iter, lambda_constraint)

    def ftf_func(a):
        iter, loss, new_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint = a
        #print(iter, "no gradient step possible in outer loop, end")
        return (max_iteration, loss, new_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint)
    
    def ft_func(a):
        iter, loss, new_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint = a
        pred = constraintsFulfilled(alpha, km, dkm, jac)  #iter > last_outer_loop_increase + 1
        a = jax.lax.cond(pred, ftt_func, ftf_func, a)
        return a

    def ff_func(a):
        #print("nothing")
        return a


    def f_func(a):
        iter, loss, new_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint = a
        pred = abs(loss - new_loss) < loop_loss_reduction
        a = jax.lax.cond(pred, ft_func, ff_func, a)
        return a

    
    def bls_more(a):
        n_alpha_grad, alpha, bls_lr, j, outer_loop_iter, lambda_constraint = a
        bls_lr *= bls_beta_minus
        return (n_alpha_grad, alpha, bls_lr, j+1, outer_loop_iter, lambda_constraint)
    
    def bls_break(a):
        n_alpha_grad, alpha, bls_lr, j, outer_loop_iter, lambda_constraint = a
        alpha = (1 - lambda_reg * bls_lr) * alpha - bls_lr * n_alpha_grad
        bls_lr = bls_lr * bls_beta_plus
        return (n_alpha_grad, alpha, bls_lr, bls_max_iter, outer_loop_iter, lambda_constraint)

    
    def body_fun(a):
        global data, losses, aux, u_loss
        iter, last_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint = a

        loss = l(alpha, km, dkm, jac)

        if iter != 0:
            iter = iter.item()
        print(iter, outer_loop_iter, loss, lambda_constraint)
        data[iter] = alpha
        losses[iter] = loss
        aux[iter] = 0
        u_loss[iter] = compute_trajectory_obstacle_cost(evaluate(alpha, km, jac))


        alpha_grad = g(alpha, km, dkm, jac)
        n_alpha_grad = alpha_grad / jnp.linalg.norm(alpha_grad) # normalized

        j = 0
        while j < bls_max_iter:
            new_alpha = (1 - lambda_reg * bls_lr) * alpha - bls_lr * n_alpha_grad
            new_loss = l(new_alpha, km, dkm, jac)
            required_loss = loss - bls_alpha * bls_lr * jnp.sum(alpha_grad * n_alpha_grad)
            print(" bls_iter", j, "bls_lr", bls_lr, "loss", new_loss, "req loss", required_loss)

            a = jax.lax.cond(new_loss > required_loss, bls_more, bls_break, (n_alpha_grad, alpha, bls_lr, j, outer_loop_iter, lambda_constraint))
            n_alpha_grad, alpha, bls_lr, j, outer_loop_iter, lambda_constraint = a
            
        pred = constraintsFulfilled(alpha, km, dkm, jac) and (outer_loop_iter > 0 or abs(loss - new_loss) < loop_loss_reduction)
            
        a = (iter, loss, new_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint)
        a = jax.lax.cond(pred, t_func, f_func, a)
        iter, loss, new_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint = a
        return (iter+1, last_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint)


    def cond_fun(a):
        iter, last_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint = a
        return iter < max_iteration #and compute_trajectory_cost(alpha, km, jac) < last_loss - loop_loss_reduction
    
    
    
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    iter, last_loss, alpha, bls_lr, outer_loop_iter, lambda_constraint = val


    #a = jax.lax.while_loop(cond_fun, body_fun, init_val)
    #iter, last_loss, alpha, bls_lr = a



et = time.time()
print("took", 1000*(et-st), "ms")


np_trajectory = np.array(evaluate(alpha, km, jac))
np.savetxt("trajectory_result.txt", np_trajectory)
#create_animation(km, dkm, jac, data, losses, aux, u_loss)



