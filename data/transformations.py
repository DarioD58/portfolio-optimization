import pandas as pd

def calculate_sma(data: pd.DataFrame, column: str, timeperiods: list) -> pd.DataFrame:
    for timeperiod in timeperiods:
        data[f"sma_{column}_{timeperiod}"] = data.groupby('symbol')[column].rolling(timeperiod).mean().reset_index(level=0, drop=True)
    return data

def calculate_ema(data: pd.DataFrame, column: str, timeperiods: list) -> pd.DataFrame:
    for timeperiod in timeperiods:
        data[f"ema_{column}_{timeperiod}"] = data.groupby('symbol')[column].ewm(span=timeperiod, adjust=False).mean().reset_index(level=0, drop=True)
    return data

def calculate_rsi(data: pd.DataFrame, column: str, timeperiods: list) -> pd.DataFrame:
    for timeperiod in timeperiods:
        delta = data.groupby('symbol')[column].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        average_gain = up.rolling(timeperiod).mean()
        average_loss = down.rolling(timeperiod).mean()
        relative_strength = average_gain / average_loss
        rsi = 100 - (100 / (1 + relative_strength))
        data[f"rsi_{column}_{timeperiod}"] = rsi
    return data

def calculate_macd(data: pd.DataFrame, column: str, timeperiods: list) -> pd.DataFrame:
    for fast_period, slow_period, signal_period in timeperiods:
        ema_fast = data.groupby('symbol')[column].ewm(span=fast_period, adjust=False).mean().reset_index(level=0, drop=True)
        ema_slow = data.groupby('symbol')[column].ewm(span=slow_period, adjust=False).mean().reset_index(level=0, drop=True)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean().reset_index(level=0, drop=True)
        histogram = macd_line - signal_line
        data[f"macd_{column}_{fast_period}_{slow_period}_{signal_period}"] = macd_line
        data[f"macdsignal_{column}_{fast_period}_{slow_period}_{signal_period}"] = signal_line
    return data, histogram

def calculate_rolling_high_low(data: pd.DataFrame, column: str, timeperiods: list) -> pd.DataFrame:
    for timeperiod in timeperiods:
        data[f'rolling_high_{column}_{timeperiod}'] = data.groupby('symbol')[column].rolling(timeperiod).max().reset_index(level=0, drop=True)
        data[f'rolling_low_{column}_{timeperiod}'] = data.groupby('symbol')[column].rolling(timeperiod).min().reset_index(level=0, drop=True)
    return data


def extract_month_quarter(data: pd.DataFrame, column: str) -> pd.DataFrame:
    data['month'] = data[column].dt.month
    data['quarter'] = data[column].dt.quarter
    return data




def label_performance(data: pd.DataFrame, column: str) -> pd.DataFrame:
    data['label'] = ''

    # Calculate monthly performance as percentage difference
    data['performance'] = data.groupby('symbol')[column].pct_change(periods=1).shift(-1)

    # Label top and bottom half performers for each month
    for date, group in data.groupby('timestamp'):
        # Sort the group by performance in descending order
        sorted_group = group.sort_values('performance', ascending=False)

        # Determine the number of top half performers
        top_half_count = int(len(sorted_group) * 0.5)

        # Assign labels to top and bottom half performers
        data.loc[sorted_group.iloc[:top_half_count].index, 'label'] = 1
        data.loc[sorted_group.iloc[top_half_count:].index, 'label'] = 0


    return data