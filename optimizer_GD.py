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



class GradientDescentOptimizer:
    def __init__(self, earlyStopping, jitLoop, dualOptimization):
        
        self.max_iteration = 200

        self.max_inner_iteration = 200
        self.max_outer_iteration = 5

        self.earlyStopping = earlyStopping
        self.jitLoop = jitLoop
        self.dualOptimization = dualOptimization

        self.lambda_constraint_increase = 10

        self.lr = 2e-3
        self.dual_lr = [2e-3, 1e-4, 1e-5, 1e-5, 1e-5]

        self.loop_loss_reduction = 1e-3
        self.dual_loop_loss_reduction = [1e-3, 1e-4, 1e-4, 1e-4, 1e-4]

        self.lambda_reg = 0.0001
        self.lambda_constraint = 1
        self.lambda_2_constraint = 0.2
        self.lambda_max_cost = 0.5

        self.trajectory = Trajectory()

        # compile loss and gradient function
        t1 = time.time()
        _ = self.trajectory.compute_trajectory_cost(self.trajectory.alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
        _ = self.trajectory.compute_trajectory_cost_g(self.trajectory.alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
        _ = self.trajectory.constraintsFulfilled(self.trajectory.alpha)
        t2 = time.time()
        print("setup object, jit-compile took", 1000*(t2-t1), "ms")

    def optimize(self, alpha):
        if self.jitLoop:
            return self.jit_optimize(alpha)
        elif self.dualOptimization:
            return self.dual_optimize(alpha)
        else:
            return self.plain_optimize(alpha)


    @partial(jax.jit, static_argnames=['self'])
    def jit_optimize(self, alpha):

        @partial(jax.jit, static_argnames=[])
        def use_newAlpha(a):
            return a[0], a[0], a[2]
        
        @partial(jax.jit, static_argnames=[])
        def use_oldAlpha(a):
            return a[1], a[1], a[2]


        @partial(jax.jit, static_argnames=[])
        def body_fun_early_stopping(iter, fa):
            alpha, last_loss = fa
            alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            new_alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
            loss = self.trajectory.compute_trajectory_cost(new_alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            (alpha, _, loss) = jax.lax.cond(loss < last_loss, use_newAlpha, use_oldAlpha, (new_alpha, alpha, loss))
            return alpha, loss


        @partial(jax.jit, static_argnames=[])
        def body_fun(iter, alpha):
            #loss = self.trajectory.compute_trajectory_cost(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
            return alpha
        
        if self.earlyStopping:
            alpha, _ = jax.lax.fori_loop(0, self.max_iteration, body_fun_early_stopping, (alpha, 1000))
        else:
            alpha = jax.lax.fori_loop(0, self.max_iteration, body_fun, alpha)

        return alpha
        

    def plain_optimize(self, alpha):

        if self.earlyStopping:
            last_loss = self.trajectory.compute_trajectory_cost(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
            for iter in range(self.max_iteration):
                alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)

                new_alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
                
                loss = self.trajectory.compute_trajectory_cost(new_alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)

                if last_loss - loss < self.loop_loss_reduction:
                    print("break after", iter, "iteration")
                    break
                else:
                    #print(iter, loss.item())
                    last_loss = loss

                alpha = new_alpha
        else:
            for iter in range(self.max_iteration):
                alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, self.lambda_constraint, self.lambda_2_constraint, self.lambda_max_cost)
                alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
                
        return alpha
    

    def dual_optimize(self, alpha):

        lambda_constraint = self.lambda_constraint
        lambda_2_constraint = self.lambda_2_constraint

        for outer_iter in range(self.max_outer_iteration):
            last_loss = self.trajectory.compute_trajectory_cost(alpha, lambda_constraint, lambda_2_constraint, self.lambda_max_cost)
            #print("init loss", last_loss)
            for innner_iter in range(self.max_inner_iteration):

                alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, lambda_constraint, lambda_2_constraint, self.lambda_max_cost)

                new_alpha = (1 - self.lambda_reg * self.dual_lr[outer_iter]) * alpha - self.dual_lr[outer_iter] * alpha_grad
                
                loss = self.trajectory.compute_trajectory_cost(new_alpha, lambda_constraint, lambda_2_constraint, self.lambda_max_cost)

                # end of current inner minimzation
                if last_loss - loss < self.dual_loop_loss_reduction[outer_iter]:
                    #print("end of inner loop minimzation too small loss change", last_loss, loss.item())
                    last_loss = loss
                    break
                else:
                    # print(outer_iter, innner_iter, loss.item())
                    last_loss = loss

                alpha = new_alpha

            #print("end of inner loop minimzation with loss", last_loss)


            if self.trajectory.constraintsFulfilled(alpha):
                #print("constrained fulfiled and inner loop minimized, end")
                break                    
            else: 
                #print("constraints violated, new outer loop", outer_iter+1, "at inner iter", innner_iter, "increase lambda")
                lambda_constraint *= self.lambda_constraint_increase
                lambda_2_constraint *= self.lambda_constraint_increase
            
        return alpha
        
        
# conditions: mask jnp where
# vmap
# jax profile to get plot
# https://www.matrixcalculus.org/ for analytic gradient

gdo = GradientDescentOptimizer(earlyStopping=True, jitLoop=False, dualOptimization=True)

profiling = True
if profiling:
    with jax.profiler.trace("/home/simon/irm_motion_planning/jax-trace", create_perfetto_link=True):
        # Run the operations to be profiled
        st = time.time()
        n_times = 100
        for i in range(n_times):
            result_alpha = gdo.optimize(gdo.trajectory.alpha.copy())
            jax.block_until_ready(result_alpha)
        et = time.time()
        print("took", 1000*(et-st)/n_times, "ms")

else:
    st = time.time()
    n_times = 100
    for i in range(n_times):
        result_alpha = gdo.optimize(gdo.trajectory.alpha.copy())
        jax.block_until_ready(result_alpha)
    et = time.time()
    print("took", 1000*(et-st)/n_times, "ms")

result_cost = gdo.trajectory.compute_trajectory_cost(result_alpha, 0, 0, gdo.lambda_max_cost)
result_constraints = gdo.trajectory.constraintsFulfilledVerbose(result_alpha, verbose=True)
print("result cost unconstraint", result_cost, "constraint fulfiled", result_constraints)

np_trajectory = np.array(gdo.trajectory.evaluate(result_alpha, gdo.trajectory.km, gdo.trajectory.jac))
np.savetxt("gd_trajectory_result.txt", np_trajectory)
