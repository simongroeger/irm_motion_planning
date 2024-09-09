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

class GradientDescentOptimizer:
    def __init__(self):
        
        self.max_iteration = 100

        self.lr_start = 0.005
        self.lr_end =   0.0002

        self.earlyStopping = False

        self.lambda_constraint_increase = 10

        self.loop_loss_reduction = 0.001

        self.lambda_reg = 0.0001
        self.lambda_constraint_start = 0.5
        self.lambda_constraint_end = 2
        self.lambda_2_constraint = 0.1
        self.lambda_max_cost = 0.8

        self.trajectory = Trajectory()

    def lambda_constraint(self, iter):
        return self.lambda_constraint_start + (self.lambda_constraint_end - self.lambda_constraint_start) * iter / self.max_iteration

    def lr(self, iter):
        return self.lr_start + (self.lr_end - self.lr_start) * iter / self.max_iteration


    def jit_optimize(self, alpha):

        l = jax.jit(self.trajectory.compute_trajectory_cost)
        g = jax.jit(jax.grad(l))
        
        for iter in range(self.max_iteration):
            #loss = l(alpha, self.lambda_constraint(iter), self.lambda_2_constraint, self.lambda_max_cost)
            alpha_grad = g(alpha, self.lambda_constraint(iter), self.lambda_2_constraint, self.lambda_max_cost)
            alpha = (1 - self.lambda_reg * self.lr(iter)) * alpha - self.lr(iter) * alpha_grad
        
        return alpha
    

    def plain_optimize(self, alpha):

        l = jax.jit(self.trajectory.compute_trajectory_cost)
        g = jax.jit(jax.grad(l))

        if self.earlyStopping:

            last_loss = 1000
            for iter in range(self.max_iteration):
                loss = l(alpha, self.lambda_constraint(iter), self.lambda_2_constraint, self.lambda_max_cost)
                #print(iter, loss, last_loss - loss)
                if last_loss - loss < self.loop_loss_reduction:
                    break
                else:
                    last_loss = loss
                alpha_grad = g(alpha, self.lambda_constraint(iter), self.lambda_2_constraint, self.lambda_max_cost)
                alpha = (1 - self.lambda_reg * self.lr(iter)) * alpha - self.lr(iter) * alpha_grad
            
        else:
        
            for iter in range(self.max_iteration):
                #loss = l(alpha, self.lambda_constraint(iter), self.lambda_2_constraint, self.lambda_max_cost)
                alpha_grad = g(alpha, self.lambda_constraint(iter), self.lambda_2_constraint, self.lambda_max_cost)
                alpha = (1 - self.lambda_reg * self.lr(iter)) * alpha - self.lr(iter) * alpha_grad
        
        return alpha
        
        
        



gdo = GradientDescentOptimizer()
jitLoop = False

o = jax.jit(gdo.jit_optimize) if jitLoop else gdo.plain_optimize

st = time.time()
result_alpha = o(gdo.trajectory.alpha)
et = time.time()
print("took", 1000*(et-st), "ms")


np_trajectory = np.array(gdo.trajectory.evaluate(result_alpha, gdo.trajectory.km, gdo.trajectory.jac))
np.savetxt("gd_trajectory_result.txt", np_trajectory)
