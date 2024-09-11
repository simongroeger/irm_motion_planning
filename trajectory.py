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
        c = 3 * t**2 - 2 * t**3
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
    def compute_trajectory_obstacle_cost(self, trajectory, lambda_max_cost):
        f = self.env.fk(trajectory)
        return self.compute_point_cost(f, lambda_max_cost)
        

    def constraintsFulfilled(self, alpha, verbose=False):
        trajectory = self.evaluate(alpha, self.km, self.jac)
        joint_velocity = self.evaluate(alpha, self.dkm, self.jac)
        
        # start and goal position
        if not self.env.start_goal_position_constraint_fulfilled(trajectory[0], trajectory[-1]):
            if verbose:
                print("violated start goal position", jnp.linalg.norm(trajectory[0]-self.env.start_config), jnp.linalg.norm(trajectory[-1]-self.env.goal_config))
            return False
        elif verbose:
            print("ok start goal position", jnp.linalg.norm(trajectory[0]-self.env.start_config), jnp.linalg.norm(trajectory[-1]-self.env.goal_config))

        
        # start and goal velocity
        if not self.env.start_goal_velocity_constraint_fulfilled(joint_velocity[0], joint_velocity[-1]):
            if verbose:
                print("violated start goal velocity", jnp.linalg.norm(joint_velocity[0]), jnp.linalg.norm(joint_velocity[-1]))
            return False
        elif verbose:
            print("ok start goal velocity", jnp.linalg.norm(joint_velocity[0]), jnp.linalg.norm(joint_velocity[-1]))

        # joint poisiton limit
        if not self.env.joint_position_constraint(trajectory):
            if verbose:
                print("joint limit")
            return False
        
        #joint velocity limit
        if not self.env.joint_velocity_constraint(joint_velocity):
            if verbose:
                print("joint velocity")
            return False

        return True
    

    @partial(jax.jit, static_argnames=['self'])
    def start_goal_cost(self, trajectory):
        s = trajectory[0]
        g = trajectory[self.N_timesteps-1]
        loss = jnp.sum(jnp.square(s-self.env.start_config)) + jnp.sum(jnp.square(g-self.env.goal_config))
        return loss


    @partial(jax.jit, static_argnames=['self'])
    def start_goal_velocity_cost(self, joint_velocity):
        loss = jnp.sum(jnp.square(joint_velocity[0])) + jnp.sum(jnp.square(joint_velocity[-1]))
        return loss


    @partial(jax.jit, static_argnames=['self'])
    def joint_limit_cost(self, trajectory):
        loss = jnp.sum(jnp.square((trajectory - self.mean_joint_position) / self.std_joint_position)) / self.N_timesteps
        return loss


    @partial(jax.jit, static_argnames=['self'])
    def joint_velocity_limit_cost(self, joint_velocity):
        loss = jnp.sum(jnp.square(joint_velocity / self.max_joint_velocity)) / self.N_timesteps
        return loss


    @partial(jax.jit, static_argnames=['self', 'lambda_constraint', 'lambda_2_constraint', 'lambda_max_cost'])
    def compute_trajectory_cost(self, alpha, lambda_constraint, lambda_2_constraint, lambda_max_cost):
        trajectory = self.evaluate(alpha, self.km, self.jac)
        joint_velocity = self.evaluate(alpha, self.dkm, self.jac)

        toc = self.compute_trajectory_obstacle_cost(trajectory, lambda_max_cost) 
        sgpc = self.start_goal_cost(trajectory)
        sgvc = self.start_goal_velocity_cost(joint_velocity)
        jpc = 0 #self.joint_limit_cost(trajectory) if not self.joint_position_constraint(trajectory) else 0
        jvc = 0 #self.joint_velocity_limit_cost(joint_velocity) if not self.joint_velocity_constraint(joint_velocity) else 0
        return toc + lambda_constraint * (sgpc + sgvc) + lambda_2_constraint * (jpc + jvc)



        