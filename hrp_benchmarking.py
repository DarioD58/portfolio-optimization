import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimizer.hrp_functional import hierarchical_risk_parity
from portfolio_optimizer.original_hrp import  main as original_hrp
import scienceplots
import timeit
import argparse
from tqdm import tqdm

plt.style.use(['science', 'no-latex', 'grid', 'ieee'])

parser = argparse.ArgumentParser()

parser.add_argument("--n_obs", type=int, default=2520)
parser.add_argument("--size", type=int, default=500)
parser.add_argument("--n_obs_increment", type=int, default=252)
parser.add_argument("--size_increment", type=int, default=50)
parser.add_argument("--increment", type=str, required=True, choices=['n_obs', 'size'])
parser.add_argument("--n_trials", type=int, default=5, help="Number of repetitions for each experiment in the timeit repeat function")
parser.add_argument("--n_iter", type=int, default=100, help="Number of iterations for the timeit function")

args = parser.parse_args()

if __name__ == "__main__":

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
    
    results_new = []
    results_original = []
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

    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, results_new, label='Proposed implementation', marker='o', markersize=2)
    ax.plot(x, results_original, label='Original implementation', marker='o', markersize=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Best time per iteration (s)')
    ax.legend()

    plt.savefig(f"figures/benchmark_{args.increment}.png")

    # Save the results
    results_df = pd.DataFrame({'new': results_new, 'original': results_original}, index=x)
    results_df.to_csv(f"results/benchmark_{args.increment}.csv")

