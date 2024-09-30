import numpy as np
import time

import jax
import jax.numpy as jnp
from functools import partial

from optimizer_BLS import BacktrackingLineSearchOptimizer
from optimizer_GD import GradientDescentOptimizer


def main():
    useBLS = True
    if useBLS:
        optimizer = BacktrackingLineSearchOptimizer(jitLoop=True)
    else:
        optimizer = GradientDescentOptimizer(earlyStopping=True, jitLoop=False, dualOptimization=True)

    def multiple_optimizations():
        st = time.time()
        n_times = 100
        for i in range(n_times):
            result_alpha = optimizer.optimize()
            jax.block_until_ready(result_alpha)
            et = time.time()
        print("took", 1000*(et-st)/n_times, "ms")
        return result_alpha

    profiling = False
    if profiling:
        with jax.profiler.trace("/home/simon/irm_motion_planning/jax-trace", create_perfetto_link=True):
            # Run the operations to be profiled
            result_alpha = multiple_optimizations()
    else:
        result_alpha = multiple_optimizations()

    avg_result_cost = optimizer.trajectory.compute_trajectory_cost(result_alpha, optimizer.env.obstacles, optimizer.env.start_config, optimizer.env.goal_config, 0, 0, 0)
    max_result_cost = optimizer.trajectory.compute_trajectory_cost(result_alpha, optimizer.env.obstacles, optimizer.env.start_config, optimizer.env.goal_config, 0, 0, 1)
    print("result cost: ( avg", avg_result_cost, ", max", max_result_cost, "). constraint fulfiled", optimizer.trajectory.constraintsFulfilledVerbose(result_alpha, optimizer.env.start_config, optimizer.env.goal_config, verbose=True))


    np_trajectory = np.array(optimizer.trajectory.evaluate(result_alpha, optimizer.trajectory.km, optimizer.trajectory.jac))
    np.savetxt("trajectory_result.txt", np_trajectory)

if __name__ == "__main__":
    main()