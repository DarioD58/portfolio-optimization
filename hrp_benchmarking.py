import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimizer.hrp_functional import hierarchical_risk_parity, evaluate_matrix_seriation, evaluate_recursive_bisection
from portfolio_optimizer.original_hrp import  main as original_hrp, evaluate_matrix_seriation as evaluate_matrix_seriation_original, evaluate_recursive_bisection as evaluate_recursive_bisection_original
from portfolio_optimizer.plotting import plot_time_of_execution
import scienceplots
import timeit
import argparse
from tqdm import tqdm


def test_hrp_function(args: argparse.Namespace):
    results_new = []
    results_original = []

    x = []
    x_label = ''

    if args.increment == 'n_obs':
        iterator = zip(range(args.n_obs_increment, args.n_obs + args.n_obs_increment, args.n_obs_increment), [args.size_increment]*(args.n_obs//args.n_obs_increment))
        x = [i for i in range(args.n_obs_increment, args.n_obs + args.n_obs_increment, args.n_obs_increment)]
        x_label = 'Number of observations'
    elif args.increment == 'size':
        iterator = zip([args.n_obs_increment]*(args.size//args.size_increment), range(args.size_increment, args.size + args.size_increment, args.size_increment))
        x = [i for i in range(args.size_increment, args.size + args.size_increment, args.size_increment)]
        x_label = 'Number of assets'
    else:
        raise ValueError("Invalid increment argument")
    
    # Time the functional implementation
    for n_obs, size in (pbar := tqdm(iterator)):
        pbar.set_description(f"Benchmarking for n_obs: {n_obs} and size: {size}")

        data = np.random.normal(0, 1, size=(n_obs, size))
        data = pd.DataFrame(data)
        time_new = timeit.repeat(lambda: hierarchical_risk_parity(data), repeat=args.n_trials, number=args.n_iter)
        results_new.append(time_new)

        time_original = timeit.repeat(lambda: original_hrp(data), repeat=args.n_trials, number=args.n_iter)
        results_original.append(time_original)

    results_new = np.min(results_new, axis=1) / args.n_iter
    results_original = np.min(results_original, axis=1) / args.n_iter

    plot_time_of_execution(results_new, results_original, x, x_label, f"figures/benchmark_{args.increment}.png")

    # Save the results
    results_df = pd.DataFrame({'new': results_new, 'original': results_original}, index=x)
    results_df.to_csv(f"results/benchmark_{args.increment}.csv")

def test_matrix_seriation_and_recursive_bisection(args: argparse.Namespace):
    results_matrix_seriation_new = []
    results_matrix_seriation_original = []

    results_recursive_bisection_new = []
    results_recursive_bisection_original = []

    x = []
    x_label = ''

    if args.increment == 'n_obs':
        iterator = zip(range(args.n_obs_increment, args.n_obs + args.n_obs_increment, args.n_obs_increment), [args.size_increment]*(args.n_obs//args.n_obs_increment))
        x = [i for i in range(args.n_obs_increment, args.n_obs + args.n_obs_increment, args.n_obs_increment)]
        x_label = 'Number of observations'
    elif args.increment == 'size':
        iterator = zip([args.n_obs_increment]*(args.size//args.size_increment), range(args.size_increment, args.size + args.size_increment, args.size_increment))
        x = [i for i in range(args.size_increment, args.size + args.size_increment, args.size_increment)]
        x_label = 'Number of assets'
    else:
        raise ValueError("Invalid increment argument")
    
    # Time the matrix seriation implementation
    for n_obs, size in (pbar := tqdm(iterator)):
        pbar.set_description(f"Benchmarking for n_obs: {n_obs} and size: {size}")

        data = np.random.normal(0, 1, size=(n_obs, size))
        data = pd.DataFrame(data)

        time_matrix_seriation_new = evaluate_matrix_seriation(data, repeat=args.n_trials, number=args.n_iter)
        results_matrix_seriation_new.append(time_matrix_seriation_new)

        time_recursive_bisection_new = evaluate_recursive_bisection(data, repeat=args.n_trials, number=args.n_iter)
        results_recursive_bisection_new.append(time_recursive_bisection_new)

        time_matrix_seriation_original = evaluate_matrix_seriation_original(data, repeat=args.n_trials, number=args.n_iter)
        results_matrix_seriation_original.append(time_matrix_seriation_original)

        time_recursive_bisection_original = evaluate_recursive_bisection_original(data, repeat=args.n_trials, number=args.n_iter)
        results_recursive_bisection_original.append(time_recursive_bisection_original)

    results_matrix_seriation_new = np.min(results_matrix_seriation_new, axis=1) / args.n_iter
    results_recursive_bisection_new = np.min(results_recursive_bisection_new, axis=1) / args.n_iter

    results_matrix_seriation_original = np.min(results_matrix_seriation_original, axis=1) / args.n_iter
    results_recursive_bisection_original = np.min(results_recursive_bisection_original, axis=1) / args.n_iter

    plot_time_of_execution(results_matrix_seriation_new, results_matrix_seriation_original, x, x_label, f"figures/benchmark_matrix_seriation_{args.increment}.png")
    plot_time_of_execution(results_recursive_bisection_new, results_recursive_bisection_original, x, x_label, f"figures/benchmark_recursive_bisection_{args.increment}.png")

    # Save the results
    results_df_matrix_seriation = pd.DataFrame({'new': results_matrix_seriation_new, 'original': results_matrix_seriation_original}, index=x)
    results_df_matrix_seriation.to_csv(f"results/benchmark_matrix_seriation_{args.increment}.csv")

    results_df_recursive_bisection = pd.DataFrame({'new': results_recursive_bisection_new, 'original': results_recursive_bisection_original}, index=x)
    results_df_recursive_bisection.to_csv(f"results/benchmark_recursive_bisection_{args.increment}.csv")

if __name__ == "__main__":
    plt.style.use(['science', 'no-latex', 'grid', 'ieee'])

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_obs", type=int, default=2520)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--n_obs_increment", type=int, default=252)
    parser.add_argument("--size_increment", type=int, default=50)
    parser.add_argument("--increment", type=str, required=True, choices=['n_obs', 'size'])
    parser.add_argument("--n_trials", type=int, default=5, help="Number of repetitions for each experiment in the timeit repeat function")
    parser.add_argument("--n_iter", type=int, default=100, help="Number of iterations for the timeit function")
    parser.add_argument("--test", type=str, required=True, choices=['entire', 'partial', 'full'], help="Run the entire benchmarking suite or a partial one")

    args = parser.parse_args()

    if args.test == 'entire':
        test_hrp_function(args)
        test_matrix_seriation_and_recursive_bisection(args)
    elif args.test == 'partial':
        test_matrix_seriation_and_recursive_bisection(args)
    elif args.test == 'full':
        test_hrp_function(args)
    else:
        raise ValueError("Invalid test argument")