import time
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from optimizer_BLS import BacktrackingLineSearchOptimizer
from optimizer_GD import GradientDescentOptimizer


def parse_args():
    parser = argparse.ArgumentParser()

    # profiling
    parser.add_argument('--profiling', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--n-times', type=int, default=1)

    # optimizer
    parser.add_argument('--optimizer-name', choices=['gd', 'bls'], default='bls')
    parser.add_argument('--jit-loop', type=lambda x: (str(x).lower() == 'true'), default=True)

    # trajectory
    parser.add_argument('--n-timesteps', type=float, default=50)
    parser.add_argument('--rbf-variance', type=float, default=0.1)
    parser.add_argument('--jac-gaussian-mean', type=float, default=0.15)

    # minimization
    parser.add_argument('--max-inner-iteration', type=int, default=200)
    parser.add_argument('--loop-loss-reduction', type=float, default=1e-3)

    # constraint dual optimization
    parser.add_argument('--max-outer-iteration', type=int, default=10)
    parser.add_argument('--lambda-constraint-increase', type=int, default=10)

    parser.add_argument('--lambda-sg-constraint', type=float, default=0.5)
    parser.add_argument('--lambda-jl-constraint', type=float, default=0.1)

    parser.add_argument('--eps-position', type=float, default=0.01)
    parser.add_argument('--eps-velocity', type=float, default=0.01)

    # loss function
    parser.add_argument('--lambda-max-cost', type=float, default=0.8)
    parser.add_argument('--lambda-reg', type=float, default=1e-4)
    parser.add_argument('--constraint-violating-dependant-loss', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--joint-safety-limit', type=float, default=0.98)

    # bls
    parser.add_argument('--max-bls-iteration', type=int, default=20)
    parser.add_argument('--bls-lr-start', type=float, default=0.2)
    parser.add_argument('--bls-alpha', type=float, default=0.01)
    parser.add_argument('--bls-beta_plus', type=float, default=1.2)
    parser.add_argument('--bls-beta_minus', type=float, default=0.5)

    # gd
    parser.add_argument('--gd-lr', type=float, default=2e-3)
    parser.add_argument('--gd-dual-lr', type=float, nargs='+', default=[2e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8])

    # robot
    parser.add_argument('--n-joints', type=int, default=3)
    parser.add_argument('--link-length', type=float, nargs='+', default=[1.5, 0.5, 1.0])
    parser.add_argument('--max-joint-velocity', type=float, default=5)
    parser.add_argument('--max-joint-position', type=float, default=2)
    parser.add_argument('--min-joint-position', type=float, default=-1)

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if args.optimizer_name == 'bls':
        optimizer = BacktrackingLineSearchOptimizer(args)
    elif args.optimizer_name == 'gd':
        optimizer = GradientDescentOptimizer(args)
    else:
        print("FATAL: not defined optimizer", args.optimizer_name)
        exit(-1)

    def multiple_optimizations():
        st = time.time()
        for i in range(args.n_times):
            result_alpha = optimizer.optimize()
            jax.block_until_ready(result_alpha)
            et = time.time()
        print("took", 1000*(et-st)/args.n_times, "ms")
        return result_alpha

    if args.profiling:
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