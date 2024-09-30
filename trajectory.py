import numpy as np
import time
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt

import jax
import jax.numpy as jnp
from functools import partial

import environment
from robot import Robot


def rbf_kernel(x_1, x_2, rbf_var):
    return jnp.exp( - (x_1 - x_2)**2 / (2 * rbf_var**2) )


def d_rbf_kernel(x_1, x_2, rbf_var):
    return (x_1 - x_2) / (rbf_var**2) * jnp.exp( - (x_1 - x_2)**2 / (2 * rbf_var**2) ) 


class Trajectory:
    def __init__(self, args):   
        self.robot = Robot(args)

        self.rbf_var = args.rbf_variance

        self.constraint_violating_dependant_loss = args.constraint_violating_dependant_loss
        self.joint_safety_limit = args.joint_safety_limit

        self.mean_joint_position = 0.5*(self.robot.max_joint_position + self.robot.min_joint_position)
        self.std_joint_position = self.robot.max_joint_position - self.mean_joint_position

        self.N_timesteps = args.n_timesteps
        self.t = jnp.linspace(0, 1, self.N_timesteps)

        # function s.t. c(0)=0, c(1)=1, c'(0)=0, c'(1)=0, c''(0)=0, c''(1)=0
        self.c = 6 * self.t**5 - 15 * self.t**4 + 10 * self.t**3

        self.km = self.create_kernel_matrix(rbf_kernel, self.t, self.t)
        self.dkm = self.create_kernel_matrix(d_rbf_kernel, self.t, self.t)
        self.jac = jnp.eye(3) + args.jac_gaussian_mean * jax.random.normal(jax.random.PRNGKey(0), (3,3))


    def create_kernel_matrix(self, kernel_f, x, x2):
        a, b = jnp.meshgrid(x, x2)
        kernel_matrix = kernel_f(a, b, self.rbf_var)
        return kernel_matrix
    

    def create_rbf_kernel_matrix(self, kernel_f, x, x2):
        a, b = jnp.meshgrid(x, x2)
        kernel_matrix = kernel_f(a, b, self.rbf_var)
        return kernel_matrix
    

    def create_d_rbf_kernel_matrix(self, kernel_f, x, x2):
        a, b = jnp.meshgrid(x, x2)
        kernel_matrix = kernel_f(a, b, self.rbf_var)
        return kernel_matrix


    @partial(jax.jit, static_argnames=['self'])
    def evaluate(self, alpha, kernel_matrix, jac):
        return kernel_matrix @ alpha @ jac


    @partial(jax.jit, static_argnames=['self'])
    def eval_any(self, alpha, kernel_f, support_x, eval_x, jac):
        return self.evaluate(alpha, self.create_kernel_matrix(kernel_f, eval_x, support_x), jac)


    @partial(jax.jit, static_argnames=['self'])
    def initTrajectory(self, start_config, goal_config):
        # define straigh_line and fit_trajectory  
        straight_line = start_config + (goal_config - start_config) * self.c[:, jnp.newaxis]
        alpha = jnp.linalg.solve(self.km, straight_line @ jnp.linalg.inv(self.jac))
        return alpha
        

    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_point_cost(self, f, obstacles, lambda_max_cost):
        cost_v = environment.compute_cost(f, obstacles)
        
        max_cost = jnp.max(cost_v)
        avg_cost = jnp.sum(cost_v) / cost_v.shape[0]
        cost = lambda_max_cost * max_cost + (1 - lambda_max_cost) * avg_cost
        return cost
    

    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_point_cost_g(self, f, obstacles, lambda_max_cost):
        cost_v, cost_g = environment.compute_cost_vg(f, obstacles)
        t_len = cost_v.shape[0]        

        # Get index of max element in cost_v
        idx_max = jnp.argmax(cost_v)

        # Gradient w.r.t. cost_v (backpropagating through max and avg operations)
        grad_avg_cost_v = jnp.ones((1, t_len)) / t_len  # Derivative of avg function
        grad_max_cost_v = jnp.zeros((1, t_len))
        grad_max_cost_v = grad_max_cost_v.at[0, idx_max].set(1)  # Derivative of max function
    
        # Combine gradients with respect to cost_v
        grad_cost_v_combined = lambda_max_cost * grad_max_cost_v + (1 - lambda_max_cost) * grad_avg_cost_v

        # Chain rule: propagate gradients back to f
        cost_result_g = jnp.multiply(grad_cost_v_combined, cost_g)

        return cost_result_g
    
    
    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_trajectory_obstacle_cost(self, trajectory, obstacles, lambda_max_cost):
        f = self.robot.fk(trajectory)
        cost = self.compute_point_cost(f, obstacles, lambda_max_cost)
        return cost
    

    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_trajectory_obstacle_cost_g(self, trajectory, obstacles, lambda_max_cost):
        f = self.robot.fk(trajectory)
        cost_g = self.compute_point_cost_g(f, obstacles, lambda_max_cost)
        jacobian = self.robot.jacobian(trajectory)
        grad_trajectory = jnp.einsum('ij,ijk->jk', cost_g, jacobian)
        return grad_trajectory
        

    @partial(jax.jit, static_argnames=['self'])
    def constraintsFulfilled(self, alpha, start_config, goal_config):
        trajectory = self.evaluate(alpha, self.km, self.jac)
        joint_velocity = self.evaluate(alpha, self.dkm, self.jac)
        
        return jnp.logical_and(self.robot.start_goal_position_constraint_fulfilled(trajectory[0], trajectory[-1], start_config, goal_config),
            jnp.logical_and(self.robot.start_goal_velocity_constraint_fulfilled(joint_velocity[0], joint_velocity[-1]),
            jnp.logical_and(self.robot.joint_position_constraint(trajectory),
            self.robot.joint_velocity_constraint(joint_velocity))))
    
    
    def constraintsFulfilledVerbose(self, alpha, start_config, goal_config, verbose=True):
        trajectory = self.evaluate(alpha, self.km, self.jac)
        joint_velocity = self.evaluate(alpha, self.dkm, self.jac)

        result = True
        
        # start and goal position
        if not self.robot.start_goal_position_constraint_fulfilled(trajectory[0], trajectory[-1], start_config, goal_config):
            if verbose:
                print("violated start goal position", jnp.linalg.norm(trajectory[0]-start_config), jnp.linalg.norm(trajectory[-1]-goal_config))
            result = False
        elif verbose:
            print("ok start goal position", jnp.linalg.norm(trajectory[0]-start_config), jnp.linalg.norm(trajectory[-1]-goal_config))

        
        # start and goal velocity
        if not self.robot.start_goal_velocity_constraint_fulfilled(joint_velocity[0], joint_velocity[-1]):
            if verbose:
                print("violated start goal velocity", jnp.linalg.norm(joint_velocity[0]), jnp.linalg.norm(joint_velocity[-1]))
            result = False
        elif verbose:
            print("ok start goal velocity", jnp.linalg.norm(joint_velocity[0]), jnp.linalg.norm(joint_velocity[-1]))

        # joint poisiton limit
        if not self.robot.joint_position_constraint(trajectory):
            if verbose:
                print("joint limit exceeded with", trajectory.max(), trajectory.min())
            result = False
        elif verbose:
            print("ok joint limit with", trajectory.max(), trajectory.min())
        
        #joint velocity limit
        if not self.robot.joint_velocity_constraint(joint_velocity):
            if verbose:
                print("joint velocity exceeded with", jnp.abs(joint_velocity).max())
            result = False
        elif verbose:
            print("ok velocity limit with", jnp.abs(joint_velocity).max())


        return result
    

    @partial(jax.jit, static_argnames=['self'])
    def start_goal_cost(self, trajectory, start_config, goal_config):
        s = trajectory[0]
        g = trajectory[self.N_timesteps-1]
        loss = 0.5 * jnp.sum(jnp.square(s-start_config)) + 0.5 * jnp.sum(jnp.square(g-goal_config))
        return loss
    

    @partial(jax.jit, static_argnames=['self'])
    def start_goal_cost_g(self, trajectory, start_config, goal_config):
        s = trajectory[0]
        g = trajectory[self.N_timesteps-1]
        gradient = jnp.zeros_like(trajectory)
        gradient = gradient.at[0].set(s - start_config)
        gradient = gradient.at[-1].set(g - goal_config)
        return gradient


    @partial(jax.jit, static_argnames=['self'])
    def start_goal_velocity_cost(self, joint_velocity):
        loss = 0.5 * jnp.sum(jnp.square(joint_velocity[0])) + 0.5 * jnp.sum(jnp.square(joint_velocity[-1]))
        return loss
    

    @partial(jax.jit, static_argnames=['self'])
    def start_goal_velocity_cost_g(self, joint_velocity):
        gradient = jnp.zeros_like(joint_velocity)
        gradient = gradient.at[0].set(joint_velocity[0])  # Set gradient for start position
        gradient = gradient.at[-1].set(joint_velocity[-1])
        return gradient


    @partial(jax.jit, static_argnames=['self'])
    def joint_position_limit_cost(self, trajectory):
        element_wise_loss = 0.5 * jnp.square((trajectory - self.mean_joint_position) / self.std_joint_position)

        # Apply the violation mask if necessary
        if self.constraint_violating_dependant_loss:
            violation_mask_max = trajectory > self.joint_safety_limit * self.robot.max_joint_position
            violation_mask_min = trajectory < self.joint_safety_limit * self.robot.min_joint_position
            violation_mask = jnp.logical_or(violation_mask_max, violation_mask_min)
            element_wise_loss = jnp.where(violation_mask, element_wise_loss, 0.0)
            
        loss = jnp.sum(element_wise_loss) / self.N_timesteps
        return loss


    @partial(jax.jit, static_argnames=['self'])
    def joint_position_limit_cost_g(self, trajectory):
        element_wise_grad = (trajectory - self.mean_joint_position) / jnp.square(self.std_joint_position)

        # Apply the violation mask if necessary
        if self.constraint_violating_dependant_loss:
            violation_mask_max = trajectory > self.joint_safety_limit * self.robot.max_joint_position
            violation_mask_min = trajectory < self.joint_safety_limit * self.robot.min_joint_position
            violation_mask = jnp.logical_or(violation_mask_max, violation_mask_min)
            element_wise_grad = jnp.where(violation_mask, element_wise_grad, 0.0)
            
        gradient = element_wise_grad / self.N_timesteps   
        return gradient


    @partial(jax.jit, static_argnames=['self'])
    def joint_velocity_limit_cost(self, joint_velocity):
        element_wise_loss = 0.5 * jnp.square(joint_velocity / self.robot.max_joint_velocity)

        # Apply the violation mask if necessary
        if self.constraint_violating_dependant_loss:
            violation_mask = jnp.abs(joint_velocity) > self.joint_safety_limit * self.robot.max_joint_velocity
            element_wise_loss = jnp.where(violation_mask, element_wise_loss, 0.0)
            
        loss = jnp.sum(element_wise_loss) / self.N_timesteps
        return loss


    @partial(jax.jit, static_argnames=['self'])
    def joint_velocity_limit_cost_g(self, joint_velocity):
        element_wise_grad = joint_velocity / jnp.square(self.robot.max_joint_velocity)

        # Apply the violation mask if necessary
        if self.constraint_violating_dependant_loss:
            violation_mask = jnp.abs(joint_velocity) > self.joint_safety_limit * self.robot.max_joint_velocity
            element_wise_grad = jnp.where(violation_mask, element_wise_grad, 0.0)
        
        gradient = element_wise_grad / self.N_timesteps   
        return gradient


    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_trajectory_cost(self, alpha, obstacles, start_config, goal_config, lambda_sg_constraint, lambda_jl_constraint, lambda_max_cost):
        trajectory = self.evaluate(alpha, self.km, self.jac)
        joint_velocity = self.evaluate(alpha, self.dkm, self.jac)

        toc = self.compute_trajectory_obstacle_cost(trajectory, obstacles, lambda_max_cost) 
        sgpc = self.start_goal_cost(trajectory, start_config, goal_config)
        sgvc = self.start_goal_velocity_cost(joint_velocity)
        jpc = self.joint_position_limit_cost(trajectory)
        jvc = self.joint_velocity_limit_cost(joint_velocity)
        return toc + lambda_sg_constraint * (sgpc + sgvc) + lambda_jl_constraint * (jpc + jvc)
    

    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_trajectory_cost_g(self, alpha, obstacles, start_config, goal_config, lambda_sg_constraint, lambda_jl_constraint, lambda_max_cost):
        trajectory = self.evaluate(alpha, self.km, self.jac)
        joint_velocity = self.evaluate(alpha, self.dkm, self.jac)

        toc_g = self.compute_trajectory_obstacle_cost_g(trajectory, obstacles, lambda_max_cost)
        sqpc_g = self.start_goal_cost_g(trajectory, start_config, goal_config)
        sqvc_g = self.start_goal_velocity_cost_g(joint_velocity)
        jpc_g = self.joint_position_limit_cost_g(trajectory)
        jvc_g = self.joint_velocity_limit_cost_g(joint_velocity)

        cost_g = (self.km.T @ (toc_g + lambda_sg_constraint * sqpc_g + lambda_jl_constraint * jpc_g) + self.dkm.T @ (lambda_sg_constraint * sqvc_g + lambda_jl_constraint * jvc_g)) @ self.jac.T

        return cost_g



        