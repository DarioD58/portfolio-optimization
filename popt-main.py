from portfolio_optimizer import run_experiments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    final_df = pd.read_csv("dataset/final_data/gold_data.csv")
    prices_df = pd.read_csv("dataset/final_data/weekly_gold_data.csv")
    sp500_bench = pd.read_csv("dataset/info_data/S&P500-prices.csv")


    sp500_bench["timestamp"] = pd.to_datetime(sp500_bench["timestamp"])
    prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"])
    final_df["timestamp"] = pd.to_datetime(final_df["timestamp"])

    results = run_experiments(
        choice_df=final_df,
        alloc_df=prices_df,
        benchmark_df=sp500_bench,
        save_pth="experiments",
        train_periods=[60],
        n_assests=[50],
        windows=['fixed']
    )

        