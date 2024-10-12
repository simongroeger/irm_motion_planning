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
    def __init__(self, args):

        self.jitLoop = args.jit_loop
        self.dualOptimization = args.max_outer_iteration > 1

        self.max_inner_iteration = args.max_inner_iteration
        self.max_outer_iteration = args.max_outer_iteration

        self.loop_loss_reduction = args.loop_loss_reduction

        self.lambda_constraint_increase = args.lambda_constraint_increase
        self.lambda_sg_constraint = args.lambda_sg_constraint
        self.lambda_jl_constraint = args.lambda_jl_constraint

        self.lambda_max_cost = args.lambda_max_cost
        self.lambda_reg = args.lambda_reg

        self.lr = args.gd_lr
        self.dual_lr = jnp.array(args.gd_dual_lr)

        if self.max_outer_iteration != len(self.dual_lr):
            print("FATAL: max_outer_iteration and dual_lr do not match")
            exit(-1)

        self.loop_loss_reduction = args.loop_loss_reduction

        self.env = Environment()
        self.trajectory = Trajectory(args)
        
        # compile loss and gradient function
        t1 = time.time()
        self.optimize()
        t2 = time.time()
        print("setup object, jit-compile took", 1000*(t2-t1), "ms")


    def optimize(self):
        init_alpha = self.trajectory.initTrajectory(self.env.start_config, self.env.goal_config)
        if self.dualOptimization:
            if self.jitLoop:
                return self.jit_dual_optimize(init_alpha, self.env.obstacles, self.env.start_config, self.env.goal_config)
            else:
                return self.dual_optimize(init_alpha)
        else:
            if self.jitLoop:
                return self.jit_optimize(init_alpha, self.env.obstacles, self.env.start_config, self.env.goal_config)
            else:
                return self.plain_optimize(init_alpha)


    @partial(jax.jit, static_argnames=['self'])
    def jit_optimize(self, alpha, obstacles, start_config, goal_config):


        @partial(jax.jit, static_argnames=[])
        def body_fun_early_stopping(iter, fa):
            alpha, last_loss = fa
            alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, obstacles, start_config, goal_config, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)
            new_alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
            loss = self.trajectory.compute_trajectory_cost(new_alpha, obstacles, start_config, goal_config, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)

            def use_newAlpha(_):
                return new_alpha, loss
            
            def use_oldAlpha(_):
                return alpha, last_loss

            alpha, loss = jax.lax.cond(loss < last_loss, use_newAlpha, use_oldAlpha, None)
            return alpha, loss


        @partial(jax.jit, static_argnames=[])
        def body_fun(iter, alpha):
            #loss = self.trajectory.compute_trajectory_cost(alpha, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)
            alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, obstacles, start_config, goal_config, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)
            alpha = (1 - self.lambda_reg * self.lr) * alpha - self.lr * alpha_grad
            return alpha
        
        alpha, _ = jax.lax.fori_loop(0, self.max_inner_iteration, body_fun_early_stopping, (alpha, 1000))
        

        return alpha
        

    def plain_optimize(self, alpha):

        last_loss = self.trajectory.compute_trajectory_cost(alpha, self.env.obstacles, self.env.start_config, self.env.goal_config, self.lambda_sg_constraint, self.lambda_jl_constraint, self.lambda_max_cost)
        for iter in range(self.max_inner_iteration):
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
                if last_loss - loss < self.loop_loss_reduction:
                    #print("end of inner loop minimzation too small loss change", last_loss, loss.item())
                    last_loss = loss
                    break
                else:
                    # print(outer_iter, innner_iter, loss.item())
                    last_loss = loss

                alpha = new_alpha

            #print("end of inner loop minimzation", innner_iter, "with loss", last_loss)


            if self.trajectory.constraintsFulfilled(alpha, self.env.start_config, self.env.goal_config):
                #print("constrained fulfiled and inner loop minimized, end")
                break                    
            else: 
                #print("constraints violated, new outer loop", outer_iter+1, "at inner iter", innner_iter, "increase lambda")
                lambda_sg_constraint *= self.lambda_constraint_increase
                lambda_jl_constraint *= self.lambda_constraint_increase
            
        return alpha


    @partial(jax.jit, static_argnames=['self'])
    def jit_dual_optimize(self, alpha, obstacles, start_config, goal_config):

        @partial(jax.jit, static_argnames=[])
        def inner_cond_fun(inner_state):
            minimized, inner_iter, _, _, _, _, _  = inner_state
            return (inner_iter < self.max_inner_iteration) & (~minimized)
        
        @partial(jax.jit, static_argnames=[])
        def inner_body_fun(inner_state):
            minimized, inner_iter, alpha, lambda_sg_constraint, lambda_jl_constraint, lr, last_loss = inner_state

            alpha_grad = self.trajectory.compute_trajectory_cost_g(alpha, obstacles, start_config, goal_config, lambda_sg_constraint, lambda_jl_constraint, self.lambda_max_cost)
            new_alpha = (1 - self.lambda_reg * lr) * alpha - lr * alpha_grad
            new_loss = self.trajectory.compute_trajectory_cost(new_alpha, obstacles, start_config, goal_config, lambda_sg_constraint, lambda_jl_constraint, self.lambda_max_cost)
            
            def inner_more(_):
                return False, inner_iter+1, new_alpha, lambda_sg_constraint, lambda_jl_constraint, lr, new_loss
            
            def inner_break(_):
                return True, inner_iter, alpha, lambda_sg_constraint, lambda_jl_constraint, lr, last_loss
        
            inner_state = jax.lax.cond(last_loss - new_loss < self.loop_loss_reduction, inner_break, inner_more, None)
            return inner_state
        


        @partial(jax.jit, static_argnames=[])
        def outer_cond_fun(outer_state):
            constraint_fulfilled, outer_iter, _, _, _ = outer_state
            return (outer_iter < self.max_outer_iteration) & (~constraint_fulfilled)

        @partial(jax.jit, static_argnames=[])
        def outer_body_fun(outer_state):
            constraint_fulfilled, outer_iter, alpha, lambda_sg_constraint, lambda_jl_constraint = outer_state

            # Inner loop
            lr = self.dual_lr[outer_iter]
            loss = self.trajectory.compute_trajectory_cost(alpha, obstacles, start_config, goal_config, lambda_sg_constraint, lambda_jl_constraint, self.lambda_max_cost)
            inner_state = (False, 0, alpha, lambda_sg_constraint, lambda_jl_constraint, lr, loss)
            minimized, innner_iter, alpha, _, _, _, _ = jax.lax.while_loop(inner_cond_fun, inner_body_fun, inner_state)

            constraint_fulfilled = self.trajectory.constraintsFulfilled(alpha, start_config, goal_config)

            def break_fn(_):
                return True, outer_iter, alpha, lambda_sg_constraint, lambda_jl_constraint

            def continue_fn(_):
                return False, outer_iter + 1, alpha, lambda_sg_constraint * self.lambda_constraint_increase, lambda_jl_constraint * self.lambda_constraint_increase

            # Use lax.cond to determine the next state
            outer_state = jax.lax.cond(constraint_fulfilled, break_fn, continue_fn, None)
            return outer_state



        # Outer loop using jax.lax.while_loop
        outer_state =  (False, 0, alpha, self.lambda_sg_constraint, self.lambda_jl_constraint)
        constraint_fulfilled, outer_iter, alpha, _, _, = jax.lax.while_loop(outer_cond_fun, outer_body_fun, outer_state)

        return alpha
