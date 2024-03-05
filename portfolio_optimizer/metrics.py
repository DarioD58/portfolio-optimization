import pandas as pd
import numpy as np

def calc_sharpe_ratio(returns, risk_free_rate=0.01):
    # Calculate the excess returns by subtracting the risk-free rate
    excess_returns = returns - risk_free_rate

    # Calculate the average (mean) of excess returns
    avg_excess_returns = excess_returns.mean()

    # Calculate the standard deviation of returns
    std_returns = excess_returns.std()

    # Calculate the Sharpe ratio
    sharpe_ratio = avg_excess_returns / std_returns

    return sharpe_ratio * np.sqrt(len(returns))

def calc_sortino_ratio(returns, risk_free_rate=0.01):
    # Calculate the excess returns by subtracting the risk-free rate
    excess_returns = returns - risk_free_rate

    # Calculate the average (mean) of excess returns
    avg_excess_returns = excess_returns.mean()

    # Calculate the standard deviation of returns
    downside_deviation = np.sqrt(np.mean((np.minimum(excess_returns, 0))**2))

    # Calculate the Sharpe ratio
    sortino_ratio = avg_excess_returns / downside_deviation

    return sortino_ratio * np.sqrt(len(returns))

def calc_calmar_ratio(returns):
    # Calculate the cumulative returns
    cumulative_returns = (1 + returns).cumprod()

    # Calculate the maximum drawdown
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    # Calculate the Calmar ratio
    calmar_ratio = returns.mean() / abs(max_drawdown)

    return calmar_ratio * np.sqrt(len(returns))