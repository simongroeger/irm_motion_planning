import numpy as np
import time
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt

import jax
import jax.numpy as jnp
from functools import partial

from environment import Environment


def rbf_kernel(x_1, x_2, rbf_var):
    return jnp.exp( - (x_1 - x_2)**2 / (2 * rbf_var**2) )


def d_rbf_kernel(x_1, x_2, rbf_var):
    return (x_1 - x_2) / (rbf_var**2) * jnp.exp( - (x_1 - x_2)**2 / (2 * rbf_var**2) ) 


class Trajectory:
    def __init__(self):   
        self.env = Environment()

        self.N_timesteps = 50

        self.rbf_var = 0.1

        self.mean_joint_position = 0.5*(self.env.max_joint_position + self.env.min_joint_position)
        self.std_joint_position = self.env.max_joint_position - self.mean_joint_position

        self.initTrajectory()


    def create_kernel_matrix(self, kernel_f, x, x2):
        a, b = jnp.meshgrid(x, x2)
        kernel_matrix = kernel_f(a, b, self.rbf_var)
        return kernel_matrix


    @partial(jax.jit, static_argnames=['self'])
    def evaluate(self, alpha, kernel_matrix, jac):
        return kernel_matrix @ alpha @ jac


    @partial(jax.jit, static_argnames=['self'])
    def eval_any(self, alpha, kernel_f, support_x, eval_x, jac):
        return self.evaluate(alpha, self.create_kernel_matrix(kernel_f, eval_x, support_x), jac)


    def initTrajectory(self):
        t = jnp.linspace(0, 1, self.N_timesteps)
        c = 6 * t**5 - 15 * t**4 + 10*t**3
        straight_line = jnp.stack((
            self.env.start_config[0] + (self.env.goal_config[0] - self.env.start_config[0]) * c,
            self.env.start_config[1] + (self.env.goal_config[1] - self.env.start_config[1]) * c,
            self.env.start_config[2] + (self.env.goal_config[2] - self.env.start_config[2]) * c
        )).T

        self.km = self.create_kernel_matrix(rbf_kernel, t, t)
        self.dkm = self.create_kernel_matrix(d_rbf_kernel, t, t)
        self.jac = jnp.eye(3) + 0.2 * jax.random.normal(jax.random.PRNGKey(0), (3,3))

        #fit_trajectory_to_straigth_line 
        self.alpha = np.linalg.solve(self.km, straight_line @ np.linalg.inv(self.jac))
        

    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_point_cost(self, f, lambda_max_cost):
        cost_v = self.env.compute_cost(f)
        
        max_cost = jnp.max(cost_v)
        avg_cost = jnp.sum(cost_v) / cost_v.shape[0]
        cost = lambda_max_cost * max_cost + (1 - lambda_max_cost) * avg_cost
        return cost
    

    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_point_cost_vg(self, f, lambda_max_cost):
        cost_v, cost_g = self.env.compute_cost_vg(f)
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
        cost_result_v = jnp.dot(grad_cost_v_combined, cost_v.T)
        cost_result_g = jnp.multiply(grad_cost_v_combined, cost_g)

        return cost_result_v, cost_result_g
    

    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_point_cost_g(self, f, lambda_max_cost):
        cost_v, cost_g = self.env.compute_cost_vg(f)
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
    def compute_trajectory_obstacle_cost(self, trajectory, lambda_max_cost):
        f = self.env.fk(trajectory)
        cost = self.compute_point_cost(f, lambda_max_cost)
        return cost
    

    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_trajectory_obstacle_cost_vg(self, trajectory, lambda_max_cost):
        f = self.env.fk(trajectory)
        cost_v, cost_g = self.compute_point_cost_vg(f, lambda_max_cost)
        jacobian = self.env.jacobian(trajectory)
        grad_trajectory = jnp.einsum('ij,ijk->jk', cost_g, jacobian)
        return cost_v, grad_trajectory
    

    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_trajectory_obstacle_cost_g(self, trajectory, lambda_max_cost):
        f = self.env.fk(trajectory)
        cost_g = self.compute_point_cost_g(f, lambda_max_cost)
        jacobian = self.env.jacobian(trajectory)
        grad_trajectory = jnp.einsum('ij,ijk->jk', cost_g, jacobian)
        return grad_trajectory
        

    @partial(jax.jit, static_argnames=['self'])
    def constraintsFulfilled(self, alpha):
        trajectory = self.evaluate(alpha, self.km, self.jac)
        joint_velocity = self.evaluate(alpha, self.dkm, self.jac)
        
        return jnp.logical_and(self.env.start_goal_position_constraint_fulfilled(trajectory[0], trajectory[-1]),
            jnp.logical_and(self.env.start_goal_velocity_constraint_fulfilled(joint_velocity[0], joint_velocity[-1]),
            jnp.logical_and(self.env.joint_position_constraint(trajectory),
            self.env.joint_velocity_constraint(joint_velocity))))
    
    
    def constraintsFulfilledVerbose(self, alpha, verbose=True):
        trajectory = self.evaluate(alpha, self.km, self.jac)
        joint_velocity = self.evaluate(alpha, self.dkm, self.jac)

        result = True
        
        # start and goal position
        if not self.env.start_goal_position_constraint_fulfilled(trajectory[0], trajectory[-1]):
            if verbose:
                print("violated start goal position", jnp.linalg.norm(trajectory[0]-self.env.start_config), jnp.linalg.norm(trajectory[-1]-self.env.goal_config))
            result = False
        elif verbose:
            print("ok start goal position", jnp.linalg.norm(trajectory[0]-self.env.start_config), jnp.linalg.norm(trajectory[-1]-self.env.goal_config))

        
        # start and goal velocity
        if not self.env.start_goal_velocity_constraint_fulfilled(joint_velocity[0], joint_velocity[-1]):
            if verbose:
                print("violated start goal velocity", jnp.linalg.norm(joint_velocity[0]), jnp.linalg.norm(joint_velocity[-1]))
            result = False
        elif verbose:
            print("ok start goal velocity", jnp.linalg.norm(joint_velocity[0]), jnp.linalg.norm(joint_velocity[-1]))

        # joint poisiton limit
        if not self.env.joint_position_constraint(trajectory):
            if verbose:
                print("joint limit exceeded")
            result = False
        
        #joint velocity limit
        if not self.env.joint_velocity_constraint(joint_velocity):
            if verbose:
                print("joint velocity exceeded")
            result = False

        return result
    

    @partial(jax.jit, static_argnames=['self'])
    def start_goal_cost(self, trajectory):
        s = trajectory[0]
        g = trajectory[self.N_timesteps-1]
        loss = 0.5 * jnp.sum(jnp.square(s-self.env.start_config)) + 0.5 * jnp.sum(jnp.square(g-self.env.goal_config))
        return loss
    

    @partial(jax.jit, static_argnames=['self'])
    def start_goal_cost_vg(self, trajectory):
        s = trajectory[0]
        g = trajectory[self.N_timesteps-1]
        loss = jnp.sum(jnp.square(s-self.env.start_config)) + jnp.sum(jnp.square(g-self.env.goal_config))
        gradient = jnp.zeros_like(trajectory)
        gradient = gradient.at[0].set(s - self.env.start_config)
        gradient = gradient.at[-1].set(g - self.env.goal_config)
        return loss, gradient
    

    @partial(jax.jit, static_argnames=['self'])
    def start_goal_cost_g(self, trajectory):
        s = trajectory[0]
        g = trajectory[self.N_timesteps-1]
        gradient = jnp.zeros_like(trajectory)
        gradient = gradient.at[0].set(s - self.env.start_config)
        gradient = gradient.at[-1].set(g - self.env.goal_config)
        return gradient


    @partial(jax.jit, static_argnames=['self'])
    def start_goal_velocity_cost(self, joint_velocity):
        loss = 0.5 * jnp.sum(jnp.square(joint_velocity[0])) + 0.5 * jnp.sum(jnp.square(joint_velocity[-1]))
        return loss
    

    @partial(jax.jit, static_argnames=['self'])
    def start_goal_velocity_cost_vg(self, joint_velocity):
        loss = jnp.sum(jnp.square(joint_velocity[0])) + jnp.sum(jnp.square(joint_velocity[-1]))
        gradient = jnp.zeros_like(joint_velocity)
        gradient = gradient.at[0].set(joint_velocity[0])  # Set gradient for start position
        gradient = gradient.at[-1].set(joint_velocity[-1])
        return loss, gradient
    

    @partial(jax.jit, static_argnames=['self'])
    def start_goal_velocity_cost_g(self, joint_velocity):
        gradient = jnp.zeros_like(joint_velocity)
        gradient = gradient.at[0].set(joint_velocity[0])  # Set gradient for start position
        gradient = gradient.at[-1].set(joint_velocity[-1])
        return gradient


    @partial(jax.jit, static_argnames=['self'])
    def joint_limit_cost(self, trajectory):
        loss = 0.5 * jnp.sum(jnp.square((trajectory - self.mean_joint_position) / self.std_joint_position)) / self.N_timesteps
        return loss


    def joint_limit_cost_vg(self, trajectory):
        loss = jnp.sum(jnp.square((trajectory - self.mean_joint_position) / self.std_joint_position)) / self.N_timesteps
        gradient = ((trajectory - self.mean_joint_position) / jnp.square(self.std_joint_position)) / self.N_timesteps
        return loss, gradient


    def joint_limit_cost_g(self, trajectory):
        gradient = ((trajectory - self.mean_joint_position) / jnp.square(self.std_joint_position)) / self.N_timesteps
        return gradient


    @partial(jax.jit, static_argnames=['self'])
    def joint_velocity_limit_cost(self, joint_velocity):
        loss = 0.5 * jnp.sum(jnp.square(joint_velocity / self.env.max_joint_velocity)) / self.N_timesteps
        return loss
    

    @partial(jax.jit, static_argnames=['self'])
    def joint_velocity_limit_cost_vg(self, joint_velocity):
        loss = 0.5 * jnp.sum(jnp.square(joint_velocity / self.env.max_joint_velocity)) / self.N_timesteps
        gradient = ((joint_velocity) / jnp.square(self.env.max_joint_velocity)) / self.N_timesteps
        return loss, gradient


    @partial(jax.jit, static_argnames=['self'])
    def joint_velocity_limit_cost_g(self, joint_velocity):
        gradient = ((joint_velocity) / jnp.square(self.env.max_joint_velocity)) / self.N_timesteps
        return gradient


    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_trajectory_cost(self, alpha, lambda_constraint, lambda_2_constraint, lambda_max_cost):
        trajectory = self.evaluate(alpha, self.km, self.jac)
        joint_velocity = self.evaluate(alpha, self.dkm, self.jac)

        toc = self.compute_trajectory_obstacle_cost(trajectory, lambda_max_cost) 
        sgpc = self.start_goal_cost(trajectory)
        sgvc = self.start_goal_velocity_cost(joint_velocity)
        jpc = self.joint_limit_cost(trajectory)
        jvc = self.joint_velocity_limit_cost(joint_velocity)
        return toc + lambda_constraint * (sgpc + sgvc) + lambda_2_constraint * (jpc + jvc)

        
    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_trajectory_cost_vg(self, alpha, lambda_constraint, lambda_2_constraint, lambda_max_cost):
        trajectory = self.evaluate(alpha, self.km, self.jac)
        joint_velocity = self.evaluate(alpha, self.dkm, self.jac)

        toc_v, toc_g = self.compute_trajectory_obstacle_cost_vg(trajectory, lambda_max_cost)
        sgpc_v, sqpc_g = self.start_goal_cost_vg(trajectory)
        sgvc_v, sqvc_g = self.start_goal_velocity_cost_vg(joint_velocity)
        jpc_v, jpc_g = self.joint_limit_cost_vg(trajectory)
        jvc_v, jvc_g = self.joint_velocity_limit_cost_vg(joint_velocity)

        cost_v = toc_v + lambda_constraint * (sgpc_v + sgvc_v) + lambda_2_constraint * (jpc_v + jvc_v)

        cost_g = (self.km.T @ (toc_g + lambda_constraint * sqpc_g + lambda_2_constraint * jpc_g) + self.dkm.T @ (lambda_constraint * sqvc_g + lambda_2_constraint * jvc_g)) @ self.jac.T

        return cost_v, cost_g
    

    @partial(jax.jit, static_argnames=['self', 'lambda_max_cost'])
    def compute_trajectory_cost_g(self, alpha, lambda_constraint, lambda_2_constraint, lambda_max_cost):
        trajectory = self.evaluate(alpha, self.km, self.jac)
        joint_velocity = self.evaluate(alpha, self.dkm, self.jac)

        toc_g = self.compute_trajectory_obstacle_cost_g(trajectory, lambda_max_cost)
        sqpc_g = self.start_goal_cost_g(trajectory)
        sqvc_g = self.start_goal_velocity_cost_g(joint_velocity)
        jpc_g = self.joint_limit_cost_g(trajectory)
        jvc_g = self.joint_velocity_limit_cost_g(joint_velocity)

        cost_g = (self.km.T @ (toc_g + lambda_constraint * sqpc_g + lambda_2_constraint * jpc_g) + self.dkm.T @ (lambda_constraint * sqvc_g + lambda_2_constraint * jvc_g)) @ self.jac.T

        return cost_g



        