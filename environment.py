import numpy as np
import time
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt

import jax
import jax.numpy as jnp


class Environment:
    def __init__(self):   
        self.max_joint_velocity = 5
        self.min_joint_position = -1
        self.max_joint_position = 2

        self.N_joints = 3
        self.link_length = jnp.array([1.5, 1.0, 0.5])

        self.start_config = jnp.array([0.0, 0.0, 0.0])
        self.goal_config = jnp.array([1.2, 1.0, 0.3])

        self.eps_velocity = 0.05
        self.eps_distance = 0.01

        self.obstacles = jnp.array([     
                                [ 2, -3],
                                [-2, 2],
                                [3, 3],
                                [-1, -2],
                                [-2, 1],
                                [-1, -1],
                                [-2, -3],
                                [-2, 0],
                                [1, 3],
                                [3, 2],
                                [2, 3]
                            ])
            

    def fk(self, config):
        c2 = config.reshape(-1, 3)
        c = jnp.cumsum(c2,axis=1)
        pos_x = self.link_length @ jnp.cos(c).T
        pos_y = self.link_length @ jnp.sin(c).T
        pos = jnp.stack((pos_x, pos_y))
        return pos


    def fk_joint(self, config, joint_id):
        c2 = config.reshape(-1, 3)[:, :joint_id].reshape(-1, joint_id)
        c = jnp.cumsum(c2,axis=1)
        ll = self.link_length[:joint_id]
        pos_x = ll @ jnp.cos(c).T
        pos_y = ll @ jnp.sin(c).T
        pos = jnp.stack((pos_x, pos_y))
        return pos


    def jacobian(self, config):
        c2 = config.reshape(-1, 3)
        c = jnp.cumsum(c2,axis=1)
        
        x = - jnp.multiply(self.link_length, jnp.sin(c))
        reverse_cumsum_x = x + jnp.sum(x,axis=1) - jnp.cumsum(x,axis=1)

        y = jnp.multiply(self.link_length, jnp.cos(c))
        reverse_cumsum_y = y + jnp.sum(y,axis=1) - jnp.cumsum(y,axis=1)
        
        j = jnp.stack((reverse_cumsum_x, reverse_cumsum_y))
        return j


    def compute_cost(self, f):
        t_len = f.shape[1]
        o_len = self.obstacles.shape[0]
        f_expand = jnp.expand_dims(f, 2) @ jnp.ones((1, 1, o_len))
        o_expand = jnp.expand_dims(self.obstacles, 2) @ jnp.ones((1, 1, t_len))
        o_reshape = o_expand.transpose((1,2,0))

        cost_v = jnp.sum(0.8 / (0.5 + jnp.linalg.norm(f_expand - o_reshape, axis=0)), axis=1)
        return cost_v


    def start_goal_position_constraint_fulfilled(self, s, g):
        if jnp.sum(jnp.square(s-self.start_config)) > jnp.square(self.eps_distance):
            return False
        if jnp.sum(jnp.square(g-self.goal_config)) > jnp.square(self.eps_distance):
            return False
        return True


    def start_goal_velocity_constraint_fulfilled(self, vs, vg):
        if jnp.sum(jnp.square(vs)) > jnp.square(self.eps_velocity):
            return False
        if jnp.sum(jnp.square(vg)) > jnp.square(self.eps_velocity):
            return False
        return True


    def joint_position_constraint(self, trajectory):
        if trajectory.max() > self.max_joint_position:
            return False 
        if trajectory.min() < self.min_joint_position:
            return False 
        return True


    def joint_velocity_constraint(self, joint_velocity):
        if jnp.abs(joint_velocity).max() > self.max_joint_velocity:
            return False
        return True






