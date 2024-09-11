import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimizer.hrp_functional import hierarchical_risk_parity
from portfolio_optimizer.metrics import calc_sharpe_ratio
import scienceplots
import argparse
from tqdm import tqdm

plt.style.use(["science", "no-latex", "grid", "ieee"])


def plot_std(df: pd.DataFrame, filename: str, window: int = 12):
    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in df.columns:
        ax.plot(
            df.iloc[window-1:, :].index, df[col].rolling(window=window).std().dropna(), label=col
        )
    ax.set_xlabel("Time in months")
    ax.set_ylabel("Standard deviation")
    ax.legend()

    plt.savefig(f"./figures/{filename}.png")


def plot_sharpe_ratio(df: pd.DataFrame, filename: str, window: int = 12):
    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 5))
    for col in df.columns:
        ax.plot(
            df.iloc[window-1:, :].index,
            df[col].rolling(window=window).apply(lambda x: calc_sharpe_ratio(x)).dropna(),
            label=col,
        )
    ax.set_xlabel("Time in months")
    ax.set_ylabel("Sharpe ratio")
    ax.legend()

    plt.savefig(f"./figures/{filename}.png")
