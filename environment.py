import numpy as np
import time
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt

import jax
import jax.numpy as jnp
from functools import partial


class Environment:
    def __init__(self):   
        
        self.start_config = jnp.array([0.0, 0.0, 0.0])
        self.goal_config = jnp.array([1.2, 1.0, 0.3])

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
        
    
@partial(jax.jit, static_argnames=[])
def compute_cost(f, obstacles):
    # Expand and broadcast `f` and `self.obstacles` for efficient vectorized distance calculation
    f_expand = jnp.expand_dims(f, 2)  # Shape: (n_dims, t_len, 1)
    o_expand = jnp.expand_dims(obstacles.T, 1)  # Shape: (n_dims, 1, o_len)
    
    # Compute the distance between `f` and `obstacles`
    fo_dist = f_expand - o_expand  # Broadcasting will handle dimensions correctly
    f_norm = jnp.sum(jnp.square(fo_dist), axis=0)  # Sum over n_dims, shape: (t_len, o_len)
    
    cost_v = jnp.sum(0.8 / (0.5 + 0.5 * f_norm), axis=1)  # Shape: (t_len,)
    return cost_v


@partial(jax.jit, static_argnames=[])
def compute_cost_vg(f, obstacles):
    # Expand and broadcast `f` and `self.obstacles` for efficient vectorized distance calculation
    f_expand = jnp.expand_dims(f, 2)  # Shape: (n_dims, t_len, 1)
    o_expand = jnp.expand_dims(obstacles.T, 1)  # Shape: (n_dims, 1, o_len)
    
    # Compute the distance between `f` and `obstacles`
    fo_dist = f_expand - o_expand  # Broadcasting will handle dimensions correctly
    f_norm = jnp.sum(jnp.square(fo_dist), axis=0)  # Sum over n_dims, shape: (t_len, o_len)
    
    cost_v = jnp.sum(0.8 / (0.5 + 0.5 * f_norm), axis=1)  # Shape: (t_len,)
    cost_g = jnp.sum(-0.8 * fo_dist / jnp.square(0.5 + 0.5 * f_norm)[None, :, :], axis=2)  # Shape: (n_dims, t_len)
    return cost_v, cost_g


@partial(jax.jit, static_argnames=[])
def compute_cost_g(f, obstacles):
    # Expand and broadcast `f` and `self.obstacles` for efficient vectorized distance calculation
    f_expand = jnp.expand_dims(f, 2)  # Shape: (n_dims, t_len, 1)
    o_expand = jnp.expand_dims(obstacles.T, 1)  # Shape: (n_dims, 1, o_len)
    
    # Compute the distance between `f` and `obstacles`
    fo_dist = f_expand - o_expand  # Broadcasting will handle dimensions correctly
    f_norm = jnp.sum(jnp.square(fo_dist), axis=0)  # Sum over n_dims, shape: (t_len, o_len)
            
    cost_g = jnp.sum(-0.8 * fo_dist / jnp.square(0.5 + 0.5 * f_norm)[None, :, :], axis=2)  # Shape: (n_dims, t_len)
    return cost_g