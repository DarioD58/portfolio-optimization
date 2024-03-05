import pandas as pd
from portfolio_optimizer.stock_choice.NaiveChoice import NaiveChoice
import numpy as np
from data.transformations import calculate_sma

class SMAChoice(NaiveChoice):
    def __init__(self, time: int, column: str = "performance_last_month") -> None:
        self.time = time
        self.column = column
        self.tag = f"SMAChoice-{time}"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        self.final_df: pd.DataFrame = pd.concat(
            [train_df.loc[:, ["symbol", "timestamp", self.column]], 
            val_df.loc[:, ["symbol", "timestamp", self.column]]],
            ignore_index=True
        )

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        self.final_df: pd.DataFrame = pd.concat(
            [self.final_df.loc[:, ["symbol", "timestamp", self.column]], 
            test_df.loc[:, ["symbol" ,"timestamp", self.column]]],
            ignore_index=True
        )
        self.final_df["timestamp"] = pd.to_datetime(self.final_df["timestamp"])
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
        self.final_df = self.final_df.sort_values(['symbol', 'timestamp'], ascending=[True, True])
        self.final_df = calculate_sma(self.final_df, self.column, [self.time])
        return self.final_df[self.final_df["timestamp"].isin(test_df["timestamp"].unique())][f"sma_{self.column}_{self.time}"].to_numpy()