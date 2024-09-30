import numpy as np
import time

import jax
import jax.numpy as jnp
from functools import partial

from trajectory import Trajectory
from environment import Environment

jax.config.update('jax_platform_name', 'cpu')
np.set_printoptions(precision=4)

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

        self.lambda_reg = 1e-4
        self.lambda_sg_constraint = 0.5
        self.lambda_jl_constraint = 0.2
        self.lambda_max_cost = 0.5

        self.env = Environment()
        self.trajectory = Trajectory()

        # compile loss and gradient function
        t1 = time.time()
        self.optimize()
        t2 = time.time()
        print("setup object, jit-compile took", 1000*(t2-t1), "ms")


    def optimize(self):
        init_alpha = self.trajectory.initTrajectory(self.env.start_config, self.env.goal_config)
        if self.jitLoop:
            return self.jit_optimize(init_alpha, self.env.obstacles, self.env.start_config, self.env.goal_config)
        elif self.dualOptimization:
            return self.dual_optimize(init_alpha)
        else:
            return self.plain_optimize(init_alpha)


    @partial(jax.jit, static_argnames=['self'])
    def jit_optimize(self, alpha, obstacles, start_config, goal_config):

        @partial(jax.jit, static_argnames=[])
        def use_newAlpha(a):
            return a[0], a[0], a[2]
        
        @partial(jax.jit, static_argnames=[])
        def use_oldAlpha(a):
            return a[1], a[1], a[2]


        @partial(jax.jit, static_argnames=[])
        def body_fun_early_stopping(iter, fa):
            alpha, last_loss = fa
            alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, obstacles, start_config, goal_config, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)
            new_alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
            loss = self.trajectory.compute_trajectory_cost(new_alpha, obstacles, start_config, goal_config, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)
            (alpha, _, loss) = jax.lax.cond(loss < last_loss, use_newAlpha, use_oldAlpha, (new_alpha, alpha, loss))
            return alpha, loss


        @partial(jax.jit, static_argnames=[])
        def body_fun(iter, alpha):
            #loss = self.trajectory.compute_trajectory_cost(alpha, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)
            alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, obstacles, start_config, goal_config, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)
            alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
            return alpha
        
        if self.earlyStopping:
            alpha, _ = jax.lax.fori_loop(0, self.max_iteration, body_fun_early_stopping, (alpha, 1000))
        else:
            alpha = jax.lax.fori_loop(0, self.max_iteration, body_fun, alpha)

        return alpha
        

    def plain_optimize(self, alpha):

        if self.earlyStopping:
            last_loss = self.trajectory.compute_trajectory_cost(alpha, self.env.obstacles, self.env.start_config, self.env.goal_config, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)
            for iter in range(self.max_iteration):
                alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, self.env.obstacles, self.env.start_config, self.env.goal_config, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)

                new_alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
                
                loss = self.trajectory.compute_trajectory_cost(new_alpha, self.env.obstacles, self.env.start_config, self.env.goal_config, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)

                if last_loss - loss < self.loop_loss_reduction:
                    print("break after", iter, "iteration")
                    break
                else:
                    #print(iter, loss.item())
                    last_loss = loss

                alpha = new_alpha
        else:
            for iter in range(self.max_iteration):
                alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, self.env.obstacles, self.env.start_config, self.env.goal_config, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)
                alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
                
        return alpha
    

    def dual_optimize(self, alpha):

        lambda_sg_constraint = self.lambda_sg_constraint
        lambda_jl_constraint = self.lambda_jl_constraint

        for outer_iter in range(self.max_outer_iteration):
            last_loss = self.trajectory.compute_trajectory_cost(alpha, self.env.obstacles, self.env.start_config, self.env.goal_config, lambda_sg_constraint, lambda_jl_constraint, self.lambda_max_cost)
            #print("init loss", last_loss)
            for innner_iter in range(self.max_inner_iteration):

                alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, self.env.obstacles, self.env.start_config, self.env.goal_config, lambda_sg_constraint, lambda_jl_constraint, self.lambda_max_cost)

                new_alpha = (1 - self.lambda_reg * self.dual_lr[outer_iter]) * alpha - self.dual_lr[outer_iter] * alpha_grad
                
                loss = self.trajectory.compute_trajectory_cost(new_alpha, self.env.obstacles, self.env.start_config, self.env.goal_config, lambda_sg_constraint, lambda_jl_constraint, self.lambda_max_cost)

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


            if self.trajectory.constraintsFulfilled(alpha, self.env.start_config, self.env.goal_config):
                #print("constrained fulfiled and inner loop minimized, end")
                break                    
            else: 
                #print("constraints violated, new outer loop", outer_iter+1, "at inner iter", innner_iter, "increase lambda")
                lambda_sg_constraint *= self.lambda_constraint_increase
                lambda_jl_constraint *= self.lambda_constraint_increase
            
        return alpha
