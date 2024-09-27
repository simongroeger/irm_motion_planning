import numpy as np
import time
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt

import jax
import jax.numpy as jnp
from functools import partial

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation

from trajectory import Trajectory

import os
os.environ['TF_XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

np.set_printoptions(precision=4)

jax.config.update('jax_platform_name', 'cpu')


class BacktrackingLineSearchOptimizer:
    def __init__(self):
        
        self.max_inner_iteration = 100
        self.max_outer_iteration = 5

        self.lambda_constraint_increase = 10

        self.loop_loss_reduction = 0.001

        self.lambda_reg = 0.0001
        self.lambda_constraint = 0.5
        self.lambda_2_constraint = 0.1
        self.lambda_max_cost = 0.8

        self.bls_lr_start = 0.2
        self.bls_alpha = 0.01
        self.bls_beta_minus = 0.5
        self.bls_beta_plus = 1.2
        self.bls_max_iter = 20

        self.trajectory = Trajectory()

        self.loss_fn = jax.jit(self.trajectory.compute_trajectory_cost)
        self.gradient_fn = jax.jit(self.trajectory.compute_trajectory_cost_g)

        _ = self.loss_fn(self.trajectory.alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
        _ = self.gradient_fn(self.trajectory.alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)


    def optimize(self, alpha):

        for outer_iter in range(self.max_outer_iteration):
            bls_lr = self.bls_lr_start
            for innner_iter in range(self.max_inner_iteration):

                loss = self.loss_fn(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
                #print(outer_iter, innner_iter, loss.item())

                alpha_grad = self.gradient_fn(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)

                n_alpha_grad = alpha_grad / jnp.linalg.norm(alpha_grad) # normalized

                alpha_norm = jnp.sum(alpha_grad * n_alpha_grad)

                for j in range(self.bls_max_iter):
                    new_alpha = (1 - self.lambda_reg * bls_lr) * alpha - bls_lr * n_alpha_grad
                    new_loss = self.loss_fn(new_alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
                    required_loss = loss - self.bls_alpha * bls_lr * alpha_norm
                    #print(" bls_iter", j, "bls_lr", bls_lr, "loss", new_loss, "req loss", required_loss)
                    
                    if new_loss > required_loss:
                        bls_lr *= self.bls_beta_minus
                    else:
                        alpha = (1 - self.lambda_reg * bls_lr) * alpha - bls_lr * n_alpha_grad
                        bls_lr = bls_lr * self.bls_beta_plus
                        break

                # end of current inner minimzation
                if loss - new_loss < self.loop_loss_reduction:
                    #print("end of inner loop minimzation too small loss change")
                    break

            if self.trajectory.constraintsFulfilled(alpha):
                #print("constrained fulfiled and inner loop minimized, end")
                break
            elif innner_iter == 0:
                #print("no gradient step possible in outer loop, end")
                break                        
            else: 
                #print()
                print("new outer loop", outer_iter+1, "at inner iter", innner_iter, "increase lambda")
                self.lambda_constraint *= self.lambda_constraint_increase
                self.lambda_2_constraint *= self.lambda_constraint_increase
            

        return alpha


    def jit_optimize(self, alpha):
        
        @partial(jax.jit, static_argnames=[])
        def bls_more(a):
            bls_iter, bls_lr, alpha, new_alpha, loss, new_loss = a
            bls_lr *= self.bls_beta_minus
            return bls_iter, bls_lr, alpha, alpha, loss, loss
        
        @partial(jax.jit, static_argnames=[])
        def bls_break(a):
            bls_iter, bls_lr, alpha, new_alpha, loss, new_loss = a
            bls_lr = bls_lr * self.bls_beta_plus
            return self.bls_max_iter, bls_lr, new_alpha, new_alpha, new_loss, new_loss
        
        @partial(jax.jit, static_argnames=[])
        def bls_cond_fun(a):
            return a[0] < self.bls_max_iter
        
        @partial(jax.jit, static_argnames=[])
        def bls_body_fun(a):
            bls_iter, bls_lr, alpha, alpha_norm, n_alpha_grad, loss = a
            new_alpha = (1 - self.lambda_reg * bls_lr) * alpha - bls_lr * n_alpha_grad
            new_loss = self.loss_fn(new_alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            required_loss = loss - self.bls_alpha * bls_lr * alpha_norm
            bls_iter, bls_lr, alpha, _, loss, _ = jax.lax.cond(new_loss > required_loss, bls_more, bls_break, (bls_iter+1, bls_lr, alpha, new_alpha, loss, new_loss))
            return (bls_iter, bls_lr, alpha, alpha_norm, n_alpha_grad, loss)

        @partial(jax.jit, static_argnames=[])
        def inner_more(a):
            return a
        
        @partial(jax.jit, static_argnames=[])
        def inner_break(a):
            inner_iter, iter_debug, alpha, bls_lr = a
            return self.max_inner_iteration, iter_debug, alpha, bls_lr
        
        @partial(jax.jit, static_argnames=[])
        def inner_cond_fun(a):
            return a[0] < self.max_inner_iteration
        
        @partial(jax.jit, static_argnames=[])
        def inner_body_fun(a):
            inner_iter, iter_debug, alpha, bls_lr = a
            loss = self.loss_fn(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            alpha_grad = self.gradient_fn(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            n_alpha_grad = alpha_grad / jnp.linalg.norm(alpha_grad) # normalized
            alpha_norm = jnp.sum(alpha_grad * n_alpha_grad)
            bls_iter, bls_lr, alpha, _, _, new_loss = jax.lax.while_loop(bls_cond_fun, bls_body_fun, (0, bls_lr, alpha, alpha_norm, n_alpha_grad, loss))
            a = jax.lax.cond(loss - new_loss < self.loop_loss_reduction, inner_break, inner_more, (inner_iter+1, inner_iter+1, alpha, bls_lr))
            return a


        for outer_iter in range(self.max_outer_iteration):

            st = time.time()

            innner_iter, iter_debug, alpha, _ = jax.lax.while_loop(inner_cond_fun, inner_body_fun, (0, 0, alpha, self.bls_lr_start))

            et = time.time()

            constraint_fulfilled = self.trajectory.constraintsFulfilled(alpha)

            ett = time.time()
            print("loop took", 1000*(et-st), "ms, sonstraint took", 1000*(ett-et), "ms")

            if constraint_fulfilled:
                print("constrained fulfiled and inner loop minimized, end")
                break
            else: 
                print("new outer loop", outer_iter+1, "at inner iter", iter_debug, "increase lambda")
                self.lambda_constraint *= self.lambda_constraint_increase
                self.lambda_2_constraint *= self.lambda_constraint_increase
            
            

        return alpha

    




blso = BacktrackingLineSearchOptimizer()

jitLoop = False

o = blso.jit_optimize if jitLoop else blso.optimize 

st = time.time()

result_alpha = o(blso.trajectory.alpha)

et = time.time()
print("took", 1000*(et-st), "ms")

result_cost = blso.trajectory.compute_trajectory_cost(result_alpha, 0, 0, blso.lambda_max_cost)
print("result cost", result_cost, "constraint fulfiled", blso.trajectory.constraintsFulfilledVerbose(result_alpha, verbose=True))

np_trajectory = np.array(blso.trajectory.evaluate(result_alpha, blso.trajectory.km, blso.trajectory.jac))
np.savetxt("bls_trajectory_result.txt", np_trajectory)
