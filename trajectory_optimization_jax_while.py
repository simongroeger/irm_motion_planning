import numpy as np
import time
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt

import jax
import jax.numpy as jnp

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation

from functools import partial

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

lr_start = 0.0002
lr_end =   0.000001

#bls
bls_lr_start = 0.1
bls_alpha = 0.1
bls_beta_minus = 0.5
bls_beta_plus = 1.5
bls_max_iter = 100

big_iter_constraint_change = 1.5

loop_loss_reduction = 0.001

lambda_reg = 0.1
lambda_constraint = 0.5
lambda_2_constraint = 0.1
lambda_max_cost = 0.8

trajectory_max_duration = 3
max_joint_velocity = 3
min_joint_position = -1
max_joint_position = 2
mean_joint_position = 0.5*(max_joint_position + min_joint_position)
std_joint_position = max_joint_position - mean_joint_position

link_length = jnp.array([1.5, 1.0, 0.5])

start_config = jnp.array([0.0, 0.0, 0.0])
goal_config = jnp.array([1.2, 1.0, 0.3])
max_start_goal_velocity = 0.5
final_max_start_goal_velocity = 0.05
max_start_goal_distance = 0.1
final_max_start_goal_distance = 0.01

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
                    [3, 2],
                    [3, 3],
                    [3, 1],
                    [2, 1],
                    [1, 1],
                    [-1, -1],
                    [-1, 0],
                    [-1, 1],
                    [2, 2]
                    ])

obstacles = obs_1

c = 2.0 / (2 + jnp.exp(4 - 8*jnp.linspace(0, 1, N_timesteps)))
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
    return jnp.exp( - (x_1 - x_2)**2 / (2 * rbf_var**2) ) / jnp.abs(x_1 - x_2)


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
    lambda_reg = 0.02
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

    #cost_v = jnp.sum(0.8 / (0.5 + jnp.sqrt(jnp.sum(jnp.square(f_expand - o_reshape), axis=0))), axis=1)
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


def start_goal_position_constraint_fulfilled(trajectory, finalConstraint=False):
    d = final_max_start_goal_distance if finalConstraint else max_start_goal_distance
    s = trajectory[0]
    if jnp.sum(jnp.square(s-start_config)) > jnp.square(d):
        return False
    g = trajectory[N_timesteps-1]
    if jnp.sum(jnp.square(g-goal_config)) > jnp.square(d):
        return False
    return True

def start_goal_velocity_constraint_fulfilled(joint_velocity, finalConstraint=False):
    d = final_max_start_goal_velocity if finalConstraint else max_start_goal_velocity
    if jnp.sum(jnp.square(joint_velocity[0])) > jnp.square(d):
        return False
    if jnp.sum(jnp.square(joint_velocity[-1])) > jnp.square(d):
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

def constraintsFulfilled(trajectory, finalConstraint=False, verbose=False):
    joint_velocity = (trajectory[1 : ] - trajectory[ : len(trajectory)-1]) * N_timesteps / trajectory_max_duration

    # start and goal position
    if not start_goal_position_constraint_fulfilled(trajectory, finalConstraint):
        if verbose:
            print("violated start goal position", jnp.linalg.norm(trajectory[0]-start_config), jnp.linalg.norm(trajectory[-1]-goal_config))
        return False
    elif verbose:
        print("ok start goal position", jnp.linalg.norm(trajectory[0]-start_config), jnp.linalg.norm(trajectory[-1]-goal_config))

    
    # start and goal velocity
    if not start_goal_velocity_constraint_fulfilled(joint_velocity, finalConstraint):
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
    loss = jnp.sum(jnp.square(joint_velocity[0] / max_start_goal_velocity)) + jnp.sum(jnp.square(joint_velocity[-1] / max_start_goal_velocity))
    return loss

def joint_limit_cost(trajectory):
    loss = jnp.sum(jnp.square((trajectory - mean_joint_position) / std_joint_position)) / N_timesteps
    return loss

def joint_velocity_limit_cost(joint_velocity):
    loss = jnp.sum(jnp.square(joint_velocity / max_joint_velocity)) / N_timesteps
    return loss

def compute_trajectory_cost(alpha, km, jac):

    trajectory = evaluate(alpha, km, jac)
    joint_velocity = (trajectory[1 : ] - trajectory[ : len(trajectory)-1]) * N_timesteps / trajectory_max_duration

    toc = compute_trajectory_obstacle_cost(trajectory) 
    sgpc = start_goal_cost(trajectory)
    sgvc = start_goal_velocity_cost(joint_velocity)
    jpc = 0 #joint_limit_cost(trajectory) if not joint_position_constraint(trajectory) else 0
    jvc = 0 #joint_velocity_limit_cost(joint_velocity) if not joint_velocity_constraint(joint_velocity) else 0
    return toc + lambda_constraint * (sgpc + sgvc) + lambda_2_constraint * (jpc + jvc)




t, alpha, km, jac, dkm = init_trajectory()

data = {}
losses = {}
u_loss = {}
aux = {}

trajectory = evaluate(alpha, km, jac)

data[0] = trajectory
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

            data[iter] = evaluate(alpha, km, jac)
            losses[iter] = loss
            aux[iter] = 0
            u_loss[iter] = compute_trajectory_obstacle_cost(data[iter])

            if abs(last_loss - loss) < 0.002 * amount_epoch_plot:
                break

            last_loss = loss
        
        alpha_grad = g(alpha, km, jac)

        alpha = (1 - lambda_reg * lr(iter)) * alpha - lr(iter) * alpha_grad

else:

    last_big_iter_increase = -1

    init_val = (0, 1000, alpha, bls_lr_start)

    def cond_fun(a):
        iter, last_loss, alpha, bls_lr = a
        return iter < max_iteration #and compute_trajectory_cost(alpha, km, jac) < last_loss - loop_loss_reduction
    
    def body_fun(a):
        iter, last_loss, alpha, bls_lr = a

        loss = compute_trajectory_cost(alpha, km, jac)

        #if iter % amount_epoch_plot == 0: 
        #    #print(iter, loss.item())
        #
        #    data[iter] = evaluate(alpha, km, jac)
        #    losses[iter] = loss
        #    aux[iter] = iter
        #    u_loss[iter] = compute_trajectory_obstacle_cost(data[iter])
        
        alpha_grad = jax.grad(compute_trajectory_cost)(alpha, km, jac)
        n_alpha_grad = alpha_grad / jnp.linalg.norm(alpha_grad) # normalized

        print("---")
        min_loss = loss
        min_bls_lr = 0
        cf = constraintsFulfilled(evaluate(alpha, km, jac), False, True)
        ul = compute_trajectory_obstacle_cost(evaluate(alpha, km, jac))
        print("iter", iter, "loss", loss, "uloss", ul, "cf", cf)
        for j in range(bls_max_iter):
            new_alpha = (1 - lambda_reg * bls_lr) * alpha - bls_lr * n_alpha_grad
            new_loss = l(new_alpha, km, jac)
            required_loss = loss - bls_alpha * bls_lr * jnp.sum(alpha_grad * n_alpha_grad)
            cf = constraintsFulfilled(evaluate(new_alpha, km, jac))
            print(" bls_iter", j, "bls_lr", bls_lr, "loss", new_loss, "req loss", required_loss, "constraint", cf)
            if new_loss < min_loss:
                min_loss = new_loss
                min_bls_lr = bls_lr

            if new_loss > required_loss:
                bls_lr *= bls_beta_minus
            else:
                # min_bls_lr = bls_lr
                print("chose", min_bls_lr, "with", min_loss)
                alpha = (1 - lambda_reg * min_bls_lr) * alpha - min_bls_lr * n_alpha_grad
                bls_lr = min_bls_lr * bls_beta_plus
                break

        #return (iter+1, loss, alpha, bls_lr)

        # constraint dual optimization:  end of current minimzation
        if abs(loss - new_loss) < loop_loss_reduction: # or iter > last_big_iter_increase + 15:
            print("abort too small loss change")
            if True: #iter > last_big_iter_increase + 1: 
                big_iter += 1
                last_big_iter_increase = iter
                print()
                print("NEW BIG ITER; INCREASE LAMBDA", big_iter)
                lambda_constraint *= big_iter_constraint_change**2
                lambda_2_constraint *= big_iter_constraint_change
                max_start_goal_distance = max(max_start_goal_distance/big_iter_constraint_change, final_max_start_goal_distance)
                max_start_goal_velocity = max(max_start_goal_velocity/big_iter_constraint_change, final_max_start_goal_velocity)
                print("new max distances,", max_start_goal_distance, max_start_goal_velocity)
                bls_lr = bls_lr_start


        return (iter+1, loss, alpha, (bls_lr_start, bls_alpha, bls_beta_minus, bls_beta_plus, bls_max_iter))
    
    

    val = init_val
    while cond_fun(val):
        val = body_fun(val)


    #a = jax.lax.while_loop(cond_fun, body_fun, init_val)
    #iter, last_loss, alpha, bls_lr = a




et = time.time()
print("took", 1000*(et-st), "ms")


np_trajectory = np.array(evaluate(alpha, km, jac))
np.savetxt("trajectory_result.txt", np_trajectory)
create_animation(data, losses, aux, u_loss)



