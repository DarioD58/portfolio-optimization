import pandas as pd
import numpy as np

class NaiveChoice:

    def __init__(self) -> None:
        self.tag = f"NaiveChoice"

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        pass

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.array(test_df["performance_last_month"])