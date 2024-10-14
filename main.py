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

    # Profiling
    parser.add_argument('--profiling', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Enable or disable jax profiling to collect performance statistics (default: False)")
    parser.add_argument('--extended-vis', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Enable or disable extended visualization features, does not work with jit loop or profiling (default: False)")
    parser.add_argument('--n-measurements', type=int, default=1,
                        help="Number of measurements to be taken during one time measurement (default: 1)")
    parser.add_argument('--n-times', type=int, default=1,
                        help="Number of times the process is repeated to generate mean and stddev(default: 1)")

    # Optimizer Options
    parser.add_argument('--optimizer-name', choices=['gd', 'bls'], default='bls',
                        help="Choose the optimizer: 'gd' for Gradient Descent, 'bls' for Backtracking Line Search (default: 'bls')")
    parser.add_argument('--jit-loop', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Enable or disable just-in-time (JIT) compilation for the loop (default: True)")

    # Trajectory Parameters
    parser.add_argument('--n-timesteps', type=float, default=50,
                        help="Number of timesteps in the trajectory (default: 50)")
    parser.add_argument('--rbf-variance', type=float, default=0.1,
                        help="Variance parameter for Radial Basis Function (RBF) kernel (default: 0.1)")
    parser.add_argument('--jac-gaussian-mean', type=float, default=0.15,
                        help="Mean value for the Gaussian noise of the matrix J (default: 0.15)")

    # Minimization Parameters
    parser.add_argument('--max-inner-iteration', type=int, default=200,
                        help="Maximum number of iterations for the inner optimization loop (default: 200)")
    parser.add_argument('--loop-loss-reduction', type=float, default=1e-3,
                        help="Minimum loss reduction threshold for each loop iteration (default: 1e-3)")

    # Constraint Dual Optimization Parameters
    parser.add_argument('--max-outer-iteration', type=int, default=10,
                        help="Maximum number of iterations for the outer optimization loop (default: 10)")
    parser.add_argument('--lambda-constraint-increase', type=int, default=10,
                        help="Increase factor for the constraint multiplier lambda-sg-constraint and lambda-jl-constraint (default: 10)")

    parser.add_argument('--lambda-sg-constraint', type=float, default=0.5,
                        help="Initial weight for the SG (Start Goal) constraint in the optimization (default: 0.5)")
    parser.add_argument('--lambda-jl-constraint', type=float, default=0.1,
                        help="Initial weight for the JL (Joint Limit) constraint in the optimization (default: 0.1)")

    parser.add_argument('--eps-position', type=float, default=0.01,
                        help="Tolerance for start goal position constraints (default: 0.01)")
    parser.add_argument('--eps-velocity', type=float, default=0.01,
                        help="Tolerance for start goal velocity constraints (default: 0.01)")

    # Loss Function Parameters
    parser.add_argument('--lambda-max-cost', type=float, default=0.5,
                        help="Maximum cost weight in the obstacle loss function (default: 0.5)")
    parser.add_argument('--lambda-reg', type=float, default=1e-4,
                        help="Regularization weight in the gradient update (default: 1e-4)")
    parser.add_argument('--constraint-violating-dependant-loss', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Enable or disable the loss dependency on constraint violations (default: True)")
    parser.add_argument('--joint-safety-limit', type=float, default=0.98,
                        help="Safety limit for the joint positions, enables loss for constraint violations at this value, ensuring constraints are respected (default: 0.98)")

    # BLS (Backtracking Line Search) Parameters
    parser.add_argument('--max-bls-iteration', type=int, default=20,
                        help="Maximum number of iterations for the Backtracking Line Search (BLS) algorithm (default: 20)")
    parser.add_argument('--bls-lr-start', type=float, default=0.2,
                        help="Initial learning rate for BLS (default: 0.2)")
    parser.add_argument('--bls-alpha', type=float, default=0.01,
                        help="Alpha parameter for BLS, controlling sufficient decrease condition (default: 0.01)")
    parser.add_argument('--bls-beta_plus', type=float, default=1.2,
                        help="Multiplicative factor to increase the learning rate in BLS (default: 1.2)")
    parser.add_argument('--bls-beta_minus', type=float, default=0.5,
                        help="Multiplicative factor to decrease the learning rate in BLS (default: 0.5)")

    # GD (Gradient Descent) Parameters
    parser.add_argument('--gd-lr', type=float, nargs='+', default=[2e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8],
                        help="Learning rates for the dual optimizatin in Gradient Descent, provided as a list (default: [2e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8])")

    # Robot Parameters
    parser.add_argument('--n-joints', type=int, default=3,
                        help="Number of joints in the robot (default: 3)")
    parser.add_argument('--link-length', type=float, nargs='+', default=[1.5, 1.0, 0.5],
                        help="Lengths of the robot's links, provided as a list (default: [1.5, 1.0, 0.5])")
    parser.add_argument('--max-joint-velocity', type=float, default=7,
                        help="Maximum velocity for the robot's joints (default: 7)")
    parser.add_argument('--max-joint-position', type=float, default=2,
                        help="Maximum allowable position for the robot's joints (default: 2)")
    parser.add_argument('--min-joint-position', type=float, default=-1,
                        help="Minimum allowable position for the robot's joints (default: -1)")


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
        runtimes = []
        for j in range(args.n_measurements):
            st = time.time()
            for i in range(args.n_times):
                result_alpha = optimizer.optimize()
                jax.block_until_ready(result_alpha)
            et = time.time()
            runtimes.append(1000*(et-st)/args.n_times)
            print("took", 1000*(et-st)/args.n_times, "ms")
        if args.n_measurements > 1:
            print("runtimes in ms: mean", np.mean(runtimes), "stddev", np.std(runtimes))
        return result_alpha

    if args.profiling:
        with jax.profiler.trace("/home/simon/irm_motion_planning/jax-trace", create_perfetto_link=True):
            # Run the operations to be profiled
            result_alpha = multiple_optimizations()
    else:
        if args.extended_vis:
            result_alpha, p = multiple_optimizations()
        else:
            result_alpha = multiple_optimizations()

    avg_result_cost = optimizer.trajectory.compute_trajectory_cost(result_alpha, optimizer.env.obstacles, optimizer.env.start_config, optimizer.env.goal_config, 0, 0, 0)
    max_result_cost = optimizer.trajectory.compute_trajectory_cost(result_alpha, optimizer.env.obstacles, optimizer.env.start_config, optimizer.env.goal_config, 0, 0, 1)
    print("result cost: ( avg", avg_result_cost, ", max", max_result_cost, "). constraint fulfiled", optimizer.trajectory.constraintsFulfilledVerbose(result_alpha, optimizer.env.start_config, optimizer.env.goal_config, verbose=True))

    np_trajectory = np.array(optimizer.trajectory.evaluate(result_alpha, optimizer.trajectory.km, optimizer.trajectory.jac))
    #print(args.lambda_max_cost)
    #np.savetxt("trajectory_result_" + str(args.lambda_max_cost) + ".txt", np_trajectory)
    np.savetxt("trajectory_result.txt", np_trajectory)

    if args.extended_vis:
        p_np = np.array(p)
        print(p_np.shape)
        np.savetxt("trajectory_series.txt", p_np.reshape((-1, args.n_joints*args.n_timesteps)))


if __name__ == "__main__":
    main()