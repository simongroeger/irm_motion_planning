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

class GradientDescentOptimizer:
    def __init__(self):
        
        self.max_iteration = 200

        self.max_inner_iteration = 200
        self.max_outer_iteration = 5

        self.earlyStopping = True

        self.lambda_constraint_increase = 10

        self.lr = 0.002
        self.dual_lr = [0.002, 0.00002, 0.00001, 0.00001, 0.00001]


        self.loop_loss_reduction = 0.001
        self.dual_loop_loss_reduction = [0.001, 0.0001, 0.0001, 0.0001, 0.0001]

        self.lambda_reg = 0.0001
        self.lambda_constraint = 0.5
        self.lambda_2_constraint = 0.1
        self.lambda_max_cost = 0.5

        self.trajectory = Trajectory()


    def jit_optimize(self, alpha):

        l = jax.jit(self.trajectory.compute_trajectory_cost)
        g = jax.jit(jax.grad(l))

        @partial(jax.jit, static_argnames=[])
        def use_newAlpha(a):
            return a[0], a[0], a[2]
        
        @partial(jax.jit, static_argnames=[])
        def use_oldAlpha(a):
            return a[1], a[1], a[2]


        @partial(jax.jit, static_argnames=[])
        def body_fun_early_stopping(iter, fa):
            alpha, last_loss = fa
            alpha_grad = g(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            new_alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
            loss = l(new_alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            (alpha, _, loss) = jax.lax.cond(loss < last_loss, use_newAlpha, use_oldAlpha, (new_alpha, alpha, loss))
            return alpha, loss


        @partial(jax.jit, static_argnames=[])
        def body_fun(iter, alpha):
            #loss = l(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            alpha_grad = g(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
            return alpha
        
        if self.earlyStopping:
            alpha, _ = jax.lax.fori_loop(0, self.max_iteration, body_fun_early_stopping, (alpha, 1000))
        else:
            alpha = jax.lax.fori_loop(0, self.max_iteration, body_fun, alpha)

        return alpha
        

    def plain_optimize(self, alpha):

        l = jax.jit(self.trajectory.compute_trajectory_cost)
        g = jax.jit(jax.grad(l))

        last_loss = l(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
        for iter in range(self.max_iteration):
            alpha_grad = g(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)

            new_alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
            
            if self.earlyStopping:
                loss = l(new_alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)

                # end of current inner minimzation
                if last_loss - loss < self.loop_loss_reduction:
                    #print("end of inner loop minimzation too small loss change")
                    break
                else:
                    print(iter, loss.item())
                    last_loss = loss

            alpha = new_alpha

        return alpha
    
    def dual_optimize(self, alpha):

        l = jax.jit(self.trajectory.compute_trajectory_cost)
        g = jax.jit(jax.grad(l))

        for outer_iter in range(self.max_outer_iteration):
            last_loss = l(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            print("init loss", last_loss)
            for innner_iter in range(self.max_inner_iteration):

                alpha_grad = g(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)

                new_alpha = (1 - self.lambda_reg * self.dual_lr[outer_iter]) * alpha - self.dual_lr[outer_iter] * alpha_grad
                
                if self.earlyStopping:
                    loss = l(new_alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)

                    print(outer_iter, innner_iter, loss.item())

                    # end of current inner minimzation
                    if last_loss - loss < self.dual_loop_loss_reduction[outer_iter]:
                        print("end of inner loop minimzation too small loss change", last_loss, loss.item())
                        break
                    else:
                        last_loss = loss

                alpha = new_alpha




            if self.trajectory.constraintsFulfilled(alpha, verbose=True):
                print("constrained fulfiled and inner loop minimized, end")
                break
            #elif innner_iter == 0:
            #    print("no gradient step possible in outer loop, end")
            #    break                        
            else: 
                #print()
                print("new outer loop", outer_iter+1, "at inner iter", innner_iter, "increase lambda")
                self.lambda_constraint *= self.lambda_constraint_increase
                self.lambda_2_constraint *= self.lambda_constraint_increase
            
        return alpha
        
        
        



gdo = GradientDescentOptimizer()
jitLoop = False
dualOptimization = True

o = jax.jit(gdo.jit_optimize) if jitLoop else gdo.dual_optimize if dualOptimization else gdo.plain_optimize 

st = time.time()
result_alpha = o(gdo.trajectory.alpha)
et = time.time()
print("took", 1000*(et-st), "ms")

result_cost = gdo.trajectory.compute_trajectory_cost(result_alpha, gdo.lambda_constraint, gdo.lambda_2_constraint, gdo.lambda_max_cost)
print("result cost", result_cost, "constraint fulfiled", gdo.trajectory.constraintsFulfilled(result_alpha, verbose=True))

np_trajectory = np.array(gdo.trajectory.evaluate(result_alpha, gdo.trajectory.km, gdo.trajectory.jac))
np.savetxt("gd_trajectory_result.txt", np_trajectory)
