import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from portfolio_optimizer.hrp_functional import hierarchical_risk_parity
from portfolio_optimizer.plotting import plot_std, plot_sharpe_ratio
from portfolio_optimizer.metrics import calc_sharpe_ratio, calc_sortino_ratio
from dataset.info_data.chosen_stocks import chosen_stocks
import scienceplots
import argparse
from tqdm import tqdm
import os

plt.style.use(["science", "no-latex", "grid", "ieee"])

parser = argparse.ArgumentParser()
parser.add_argument(
    "--subsample",
    type=str,
    default="all",
    choices=["all", "gics", "gics-uniform", "predefined"],
    help="Subsample to use for the analysis",
)
parser.add_argument(
    "--size",
    type=int,
    default=22,
    help="Size of the subsample to use. Only used when subsample is set to 'gics-uniform'",
)
parser.add_argument(
    "--gics",
    type=str,
    default="Information Technology",
    help="GICS sector to use for the analysis. Only used when subsample is set to 'gics'",
)
parser.add_argument(
    "--lookback",
    type=int,
    default=36,
    help="Number of months to look back for the calculation of the covariance matrix",
)
parser.add_argument(
    "--use-risk-free",
    action="store_true",
    help="Use the risk-free rate for the calculation of the Sharpe ratio",
)

args = parser.parse_args()

if __name__ == "__main__":
    if os.path.exists("results") is False:
        os.makedirs("results")

    if os.path.exists("figures") is False:
        os.makedirs("figures")

    weekly_returns = pd.read_csv("dataset/final_data/weekly_gold_data.csv")
    monthly_returns = pd.read_csv("dataset/final_data/gold_data.csv")

    info_df = pd.read_csv("dataset/info_data/S&P500-Info.csv")
    risk_free_rate = pd.read_csv("dataset/final_data/DGS10.csv")
    if args.use_risk_free:
        risk_free_rate = risk_free_rate.set_index("DATE").div(100).div(12)
    else:
        risk_free_rate = risk_free_rate.set_index("DATE")
        risk_free_rate.loc[:, "DGS10"] = 0.01

    risk_free_rate.index = pd.to_datetime(risk_free_rate.index)

    monthly_returns["timestamp"] = pd.to_datetime(monthly_returns["timestamp"])

    weekly_returns["timestamp"] = pd.to_datetime(weekly_returns["timestamp"])

    monthly_returns = monthly_returns.pivot_table(
        index="timestamp", columns="symbol", values="performance_last_month"
    )
    weekly_returns = weekly_returns.pivot_table(
        index="timestamp", columns="symbol", values="performance_last_week"
    )

    if args.subsample == "all":
        pass
    elif args.subsample == "gics":
        info_df = info_df[info_df["GICS Sector"] == args.gics]
        monthly_returns = monthly_returns.loc[
            :, monthly_returns.columns.isin(info_df["Symbol"].values)
        ]
        weekly_returns = weekly_returns.loc[
            :, weekly_returns.columns.isin(info_df["Symbol"].values)
        ]
    elif args.subsample == "gics-uniform":
        all_gics = list(info_df["GICS Sector"].unique())
        chosen_stocks = (
            info_df.groupby("GICS Sector")
            .apply(lambda x: x.sample(args.size // len(all_gics), random_state=0))
            .reset_index(drop=True)
        )
        monthly_returns = monthly_returns.loc[
            :, monthly_returns.columns.isin(chosen_stocks["Symbol"].values)
        ]
        weekly_returns = weekly_returns.loc[
            :, weekly_returns.columns.isin(chosen_stocks["Symbol"].values)
        ]

        chosen_stocks.to_csv(f"results/{args.subsample}_{args.size}.csv")
    elif args.subsample == "predefined":
        monthly_returns = monthly_returns.loc[
            :, monthly_returns.columns.isin(chosen_stocks)
        ]
        weekly_returns = weekly_returns.loc[
            :, weekly_returns.columns.isin(chosen_stocks)
        ]
    else:
        raise ValueError("Invalid subsample argument")

    relevant_monthly_returns = monthly_returns[
        (monthly_returns.index >= "2005-01-01")
        & (monthly_returns.index <= "2023-01-01")
    ]

    portfolio_returns_hrp = []
    portfolio_returns_eq = []
    dates = []

    for date in tqdm(relevant_monthly_returns.index):
        relevant_tickers = relevant_monthly_returns[
            relevant_monthly_returns.index == date
        ].isna()

        data = weekly_returns.loc[
            (weekly_returns.index >= date - pd.DateOffset(months=args.lookback))
            & (weekly_returns.index <= date),
            ~relevant_tickers.values[0],
        ]

        hrp_weights = hierarchical_risk_parity(data)
        portfolio_returns_hrp.append(
            relevant_monthly_returns.loc[[date], ~relevant_tickers.values[0]]
            .mul(hrp_weights)
            .sum(axis=1)
            .values[0]
        )
        portfolio_returns_eq.append(
            relevant_monthly_returns.loc[date, ~relevant_tickers.values[0]].mean()
        )
        dates.append(date)

    final_results = pd.DataFrame(
        {
            "Date": dates,
            "HRP": portfolio_returns_hrp,
            "1/N": portfolio_returns_eq,
        }
    ).set_index("Date")

    # Risk free rate between the beginning and end of the period
    risk_free_rate = risk_free_rate.loc[
        risk_free_rate.index <= final_results.index.max(), "DGS10"
    ].iloc[-len(final_results) :]

    plot_std(
        final_results, f"{args.subsample}_{args.lookback}_std_comparison", window=12
    )
    plot_sharpe_ratio(
        final_results,
        f"{args.subsample}_{args.lookback}_sharpe_ratio_comparison",
        window=12,
        risk_free_rate=risk_free_rate,
    )

    metrics_df = pd.DataFrame(
        {
            "Mean returns": [final_results["HRP"].mean(), final_results["1/N"].mean()],
            "Standard Deviation": [
                final_results["HRP"].std(),
                final_results["1/N"].std(),
            ],
            "Sharpe Ratio": [
                calc_sharpe_ratio(final_results["HRP"], risk_free_rate=risk_free_rate),
                calc_sharpe_ratio(final_results["1/N"], risk_free_rate=risk_free_rate),
            ],
            "Sortino Ratio": [
                calc_sortino_ratio(final_results["HRP"], risk_free_rate=risk_free_rate),
                calc_sortino_ratio(final_results["1/N"], risk_free_rate=risk_free_rate),
            ],
        },
        columns=["Mean returns", "Standard Deviation", "Sharpe Ratio", "Sortino Ratio"],
        index=["HRP", "1/N"],
    )

    metrics_df.to_csv(f"results/{args.subsample}_{args.lookback}_metrics.csv")
