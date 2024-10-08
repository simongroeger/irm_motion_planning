import numpy as np
import time
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt

import jax
import jax.numpy as jnp
from functools import partial

from environment import Environment

class Robot:
    def __init__(self, args):   
        self.max_joint_velocity = args.max_joint_velocity
        self.min_joint_position = args.min_joint_position
        self.max_joint_position = args.max_joint_position

        self.N_joints = args.n_joints
        self.link_length = jnp.array(args.link_length)

        if self.N_joints != len(self.link_length):
            print("FATAL: n_joints and link_length do not match")
            exit(-1)

        self.eps_velocity = args.eps_velocity
        self.eps_distance = args.eps_position

            
    @partial(jax.jit, static_argnames=['self'])
    def fk(self, config):
        c2 = config.reshape(-1, 3)
        c = jnp.cumsum(c2,axis=1)
        pos_x = self.link_length @ jnp.cos(c).T
        pos_y = self.link_length @ jnp.sin(c).T
        pos = jnp.stack((pos_x, pos_y))
        return pos
   

    @partial(jax.jit, static_argnames=['self'])
    def fk_joint_1(self, config):
        joint_id = 1
        c2 = config.reshape(-1, 3)[:, :joint_id].reshape(-1, joint_id)
        c = jnp.cumsum(c2,axis=1)
        ll = self.link_length[:joint_id]
        pos_x = ll @ jnp.cos(c).T
        pos_y = ll @ jnp.sin(c).T
        pos = jnp.stack((pos_x, pos_y))
        return pos


    @partial(jax.jit, static_argnames=['self'])
    def fk_joint_2(self, config):
        joint_id = 2
        c2 = config.reshape(-1, 3)[:, :joint_id].reshape(-1, joint_id)
        c = jnp.cumsum(c2,axis=1)
        ll = self.link_length[:joint_id]
        pos_x = ll @ jnp.cos(c).T
        pos_y = ll @ jnp.sin(c).T
        pos = jnp.stack((pos_x, pos_y))
        return pos


    @partial(jax.jit, static_argnames=['self'])
    def fk_joint_3(self, config):
        joint_id = 3
        c2 = config.reshape(-1, 3)[:, :joint_id].reshape(-1, joint_id)
        c = jnp.cumsum(c2,axis=1)
        ll = self.link_length[:joint_id]
        pos_x = ll @ jnp.cos(c).T
        pos_y = ll @ jnp.sin(c).T
        pos = jnp.stack((pos_x, pos_y))
        return pos


    @partial(jax.jit, static_argnames=['self'])
    def jacobian(self, config):
        c2 = config.reshape(-1, 3)
        c = jnp.cumsum(c2,axis=1)

        x = - jnp.multiply(self.link_length, jnp.sin(c))
        reverse_cumsum_x = x + jnp.expand_dims(jnp.sum(x,axis=1), 1) - jnp.cumsum(x,axis=1)

        y = jnp.multiply(self.link_length, jnp.cos(c))
        reverse_cumsum_y = y + jnp.expand_dims(jnp.sum(y,axis=1), 1) - jnp.cumsum(y,axis=1)
        
        j = jnp.stack((reverse_cumsum_x, reverse_cumsum_y))
        return j


    @partial(jax.jit, static_argnames=['self'])
    def start_goal_position_constraint_fulfilled(self, s, g, start_config, goal_config):
        start_constraint = jnp.linalg.norm(s-start_config) < self.eps_distance
        goal_constraint = jnp.linalg.norm(g-goal_config) < self.eps_distance
        return jnp.logical_and(start_constraint, goal_constraint)


    @partial(jax.jit, static_argnames=['self'])
    def start_goal_velocity_constraint_fulfilled(self, vs, vg):
        start_constraint = jnp.linalg.norm(vs) < self.eps_velocity
        goal_constraint = jnp.linalg.norm(vg) < self.eps_velocity
        return jnp.logical_and(start_constraint, goal_constraint)


    @partial(jax.jit, static_argnames=['self'])
    def joint_position_constraint(self, trajectory):
        max_constraint = trajectory.max() <= self.max_joint_position
        min_constraint = trajectory.min() >= self.min_joint_position
        return jnp.logical_and(max_constraint, min_constraint)


    @partial(jax.jit, static_argnames=['self'])
    def joint_velocity_constraint(self, joint_velocity):
        return jnp.abs(joint_velocity).max() <= self.max_joint_velocity


