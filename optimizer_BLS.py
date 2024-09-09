import numpy as np
import time
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt

import jax
import jax.numpy as jnp

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

    def optimize(self, alpha):
        trajectory = Trajectory()

        l = jax.jit(trajectory.compute_trajectory_cost)
        g = jax.jit(jax.grad(l))

        for outer_iter in range(self.max_outer_iteration):
            bls_lr = self.bls_lr_start
            for innner_iter in range(self.max_inner_iteration):

                loss = l(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
                #print(outer_iter, innner_iter, loss.item())

                alpha_grad = g(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)

                n_alpha_grad = alpha_grad / jnp.linalg.norm(alpha_grad) # normalized

                for j in range(self.bls_max_iter):
                    new_alpha = (1 - self.lambda_reg * bls_lr) * alpha - bls_lr * n_alpha_grad
                    new_loss = l(new_alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
                    required_loss = loss - self.bls_alpha * bls_lr * jnp.sum(alpha_grad * n_alpha_grad)
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

            if self.trajectory.constraintsFulfilled(alpha, verbose=False):
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


    




blso = BacktrackingLineSearchOptimizer()

st = time.time()

result_alpha = blso.optimize(blso.trajectory.alpha)

et = time.time()
print("took", 1000*(et-st), "ms")

np_trajectory = np.array(blso.trajectory.evaluate(result_alpha, blso.trajectory.km, blso.trajectory.jac))
np.savetxt("bls_trajectory_result.txt", np_trajectory)
