import pandas as pd

class SimpleAllocator:
    def __init__(self) -> None:
        self.tag = f"SimpleAllocator"

    def fit(self, df: pd.DataFrame) -> dict:
        return {tick: 1/len(df.columns) for tick in df.columns}