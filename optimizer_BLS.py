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
from environment import Environment

np.set_printoptions(precision=4)

jax.config.update('jax_platform_name', 'cpu')


class BacktrackingLineSearchOptimizer:
    def __init__(self, jitLoop):
        
        self.max_inner_iteration = 200
        self.max_outer_iteration = 10

        self.lambda_constraint_increase = 10

        self.loop_loss_reduction = 0.001

        self.lambda_reg = 1e-4
        self.lambda_sg_constraint = 0.5
        self.lambda_jl_constraint = 0.2
        self.lambda_max_cost = 0.8

        self.bls_lr_start = 0.2
        self.bls_alpha = 0.01
        self.bls_beta_minus = 0.5
        self.bls_beta_plus = 1.2
        self.bls_max_iter = 20

        self.jitLoop = jitLoop

        self.env = Environment()
        self.trajectory = Trajectory()

        # compile optimization incl loss and gradient function
        t1 = time.time()
        _ = self.optimize()
        t2 = time.time()
        print("setup object, jit-compile took", 1000*(t2-t1), "ms")


    def optimize(self):
        init_alpha = self.trajectory.initTrajectory(self.env.start_config, self.env.goal_config)
        if self.jitLoop:
            return self.jit_optimize(init_alpha, self.env.obstacles, self.env.start_config, self.env.goal_config)
        else:
            return self.plain_optimize(init_alpha)


    def plain_optimize(self, alpha):

        lambda_sg_constraint = self.lambda_sg_constraint
        lambda_jl_constraint = self.lambda_jl_constraint

        for outer_iter in range(self.max_outer_iteration):
            bls_lr = self.bls_lr_start
            t1 = time.time()

            for innner_iter in range(self.max_inner_iteration):

                loss = self.trajectory.compute_trajectory_cost(alpha, self.env.obstacles, self.env.start_config, self.env.goal_config, lambda_sg_constraint, lambda_jl_constraint, self.lambda_max_cost)
                # print(outer_iter, innner_iter, loss.item())

                alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, self.env.obstacles, self.env.start_config, self.env.goal_config, lambda_sg_constraint, lambda_jl_constraint, self.lambda_max_cost)

                n_alpha_grad = alpha_grad / jnp.linalg.norm(alpha_grad) # normalized

                alpha_norm = jnp.sum(alpha_grad * n_alpha_grad)

                for j in range(self.bls_max_iter):
                    new_alpha = (1 - self.lambda_reg * bls_lr) * alpha - bls_lr * n_alpha_grad
                    new_loss = self.trajectory.compute_trajectory_cost(new_alpha, self.env.obstacles, self.env.start_config, self.env.goal_config, lambda_sg_constraint, lambda_jl_constraint, self.lambda_max_cost)
                    required_loss = loss - self.bls_alpha * bls_lr * alpha_norm
                    # print(" bls_iter", j, "bls_lr", bls_lr, "loss", new_loss, "req loss", required_loss)
                    
                    if new_loss > required_loss:
                        bls_lr *= self.bls_beta_minus
                    else:
                        alpha = new_alpha
                        bls_lr = bls_lr * self.bls_beta_plus
                        break

                # end of current inner minimzation
                if loss - new_loss < self.loop_loss_reduction:
                    #print("end of inner loop minimzation too small loss change")
                    break

            print("end of inner loop minimzation", outer_iter, "at inner iter", innner_iter, "with loss", loss)

            t2 = time.time()

            if self.trajectory.constraintsFulfilled(alpha, self.env.start_config, self.env.goal_config):
                print("constrained fulfiled and inner loop minimized, end")
                break                    
            else: 
                print("constraints violated, new outer loop", outer_iter+1, "increase lambda")
                lambda_sg_constraint *= self.lambda_constraint_increase
                lambda_jl_constraint *= self.lambda_constraint_increase

            t3 = time.time()
            print("loop took", 1000*(t2-t1), "ms, sonstraint took", 1000*(t3-t2), "ms")

        return alpha

    @partial(jax.jit, static_argnames=['self'])
    def jit_optimize(self, alpha, obstacles, start_config, goal_config):
    

        @partial(jax.jit, static_argnames=[])
        def bls_cond_fun(bls_state):
            obtained_required_loss, bls_iter, _, _, _, _, _, _, _ = bls_state
            return (bls_iter < self.bls_max_iter) & (~obtained_required_loss)
        
        @partial(jax.jit, static_argnames=[])
        def bls_body_fun(bls_state):
            obtained_required_loss, bls_iter, bls_lr, alpha, alpha_norm, n_alpha_grad, loss, lambda_sg_constraint, lambda_jl_constraint = bls_state

            new_alpha = (1 - self.lambda_reg * bls_lr) * alpha - bls_lr * n_alpha_grad
            new_loss = self.trajectory.compute_trajectory_cost(new_alpha, obstacles, start_config, goal_config, lambda_sg_constraint, lambda_jl_constraint, self.lambda_max_cost)
            required_loss = loss - self.bls_alpha * bls_lr * alpha_norm

            def bls_more(_):
                return False, bls_iter + 1, bls_lr * self.bls_beta_minus, alpha, alpha_norm, n_alpha_grad, loss, lambda_sg_constraint, lambda_jl_constraint
            
            def bls_break(_):
                return True, bls_iter, bls_lr * self.bls_beta_plus, new_alpha, alpha_norm, n_alpha_grad, new_loss, lambda_sg_constraint, lambda_jl_constraint

            bls_state = jax.lax.cond(new_loss > required_loss, bls_more, bls_break, None)
            return bls_state



        @partial(jax.jit, static_argnames=[])
        def inner_cond_fun(inner_state):
            minimized, inner_iter, _, _, _, _,  = inner_state
            return (inner_iter < self.max_inner_iteration) & (~minimized)
        
        @partial(jax.jit, static_argnames=[])
        def inner_body_fun(inner_state):
            minimized, inner_iter, alpha, bls_lr, lambda_sg_constraint, lambda_jl_constraint = inner_state

            loss = self.trajectory.compute_trajectory_cost(alpha, obstacles, start_config, goal_config, lambda_sg_constraint, lambda_jl_constraint, self.lambda_max_cost)
            alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, obstacles, start_config, goal_config, lambda_sg_constraint, lambda_jl_constraint, self.lambda_max_cost)
            n_alpha_grad = alpha_grad / jnp.linalg.norm(alpha_grad) # normalized
            alpha_norm = jnp.sum(alpha_grad * n_alpha_grad)

            # bls loop
            bls_state = (False, 0, bls_lr, alpha, alpha_norm, n_alpha_grad, loss, lambda_sg_constraint, lambda_jl_constraint)
            obtained_required_loss, bls_iter, bls_lr, alpha, _, _, new_loss, _, _  = jax.lax.while_loop(bls_cond_fun, bls_body_fun, bls_state)

            def inner_more(_):
                return False, inner_iter+1, alpha, bls_lr, lambda_sg_constraint, lambda_jl_constraint
            
            def inner_break(_):
                return True, inner_iter, alpha, bls_lr, lambda_sg_constraint, lambda_jl_constraint
        
            inner_state = jax.lax.cond(loss - new_loss < self.loop_loss_reduction, inner_break, inner_more, None)
            return inner_state
        


        @partial(jax.jit, static_argnames=[])
        def outer_cond_fun(state):
            constraint_fulfilled, outer_iter, _, _, _ = state
            return (outer_iter < self.max_outer_iteration) & (~constraint_fulfilled)

        @partial(jax.jit, static_argnames=[])
        def outer_body_fun(state):
            constraint_fulfilled, outer_iter, alpha, lambda_sg_constraint, lambda_jl_constraint = state

            # Inner loop
            inner_state = (False, 0, alpha, self.bls_lr_start, lambda_sg_constraint, lambda_jl_constraint)
            minimized, innner_iter, alpha, _, _, _ = jax.lax.while_loop(inner_cond_fun, inner_body_fun, inner_state)

            constraint_fulfilled = self.trajectory.constraintsFulfilled(alpha, start_config, goal_config)

            def break_fn(_):
                return True, outer_iter, alpha, lambda_sg_constraint, lambda_jl_constraint

            def continue_fn(_):
                return False, outer_iter + 1, alpha, lambda_sg_constraint * self.lambda_constraint_increase, lambda_jl_constraint * self.lambda_constraint_increase

            # Use lax.cond to determine the next state
            state = jax.lax.cond(constraint_fulfilled, break_fn, continue_fn, None)

            return state


        # Outer loop using jax.lax.while_loop
        constraint_fulfilled, outer_iter, alpha, _, _, = jax.lax.while_loop(outer_cond_fun, outer_body_fun, (False, 0, alpha, self.lambda_sg_constraint, self.lambda_jl_constraint))

        return alpha
