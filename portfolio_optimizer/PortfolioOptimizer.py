import pandas as pd
from portfolio_optimizer.weight_allocators.SimpleAllocator import SimpleAllocator
from portfolio_optimizer.stock_choice.NaiveChoice import NaiveChoice
from tqdm import trange

class PortfolioOptimizer:

    def __init__(
            self, 
            historical_data: pd.DataFrame,
            allocator: SimpleAllocator,
            picker: NaiveChoice,
            benchmark: pd.DataFrame,
            train_period: int,
            allocation_period: int = 24,
            validation_period: int = 24,
            test_period: int = 120,
            past_performance: pd.DataFrame = None,
            window: str = 'fixed',
            retrain_period: int = 12,
            verbose: bool = False
        ) -> None:

        self.historical_data = historical_data
        if past_performance is not None:
            self.past_performance = past_performance.pivot(index='timestamp', columns='symbol', values='performance_last_month')
        else:
            self.past_performance = historical_data.pivot(index='timestamp', columns='symbol', values='performance_last_month')
        self.benchmark = benchmark
        self.benchmark['performance'] = self.benchmark.sort_values(["timestamp"])["adjusted close"].pct_change(periods=1).shift(-1)

        self.window = window
        self.num_periods = int(test_period/retrain_period) if self.window != 'fixed' else 1
        self.train_period = train_period
        self.validation_period = validation_period
        self.retrain_period = retrain_period if self.window != 'fixed' else 0
        self.test_period = retrain_period if self.window != 'fixed' else test_period
        self.testing_window = test_period
        self.allocation_period = allocation_period

        self.allocator = allocator
        self.picker = picker

        self.verbose = verbose


    def _make_dataset(self, iteration: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.verbose:
            print(f"Iteration: {iteration}")

        if self.window == 'rolling':
            train_df, val_df, test_df = self._dataset_split(self.testing_window - (self.retrain_period * (iteration + 1)))
        elif self.window == 'expanding':
            self.train_period += self.retrain_period
            train_df, val_df, test_df = self._dataset_split(self.testing_window - (self.retrain_period * (iteration + 1)))
        elif self.window == 'fixed':
            train_df, val_df, test_df = self._dataset_split()
        else:
            raise NotImplementedError
        
        return train_df, val_df, test_df



    def _dataset_split(self, offset: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        offset_end_date = self.historical_data['timestamp'].max()
        offset_start_date = offset_end_date - pd.DateOffset(months=offset)
        test_end_date = offset_start_date - (pd.DateOffset(months=1) if offset > 0 else pd.DateOffset(months=0))
        test_start_date = test_end_date - pd.DateOffset(months=self.test_period) + (pd.DateOffset(months=1) if offset > 0 else pd.DateOffset(months=0))
        validation_end_date = test_start_date - pd.DateOffset(months=1)
        validation_start_date = validation_end_date - pd.DateOffset(months=self.validation_period)
        train_end_date = validation_start_date - pd.DateOffset(months=1)
        if self.train_period:
            train_start_date = train_end_date - pd.DateOffset(months=self.train_period)
        else:
            train_start_date = self.historical_data['timestamp'].min()

        train_df = self.historical_data[(self.historical_data['timestamp'] >= train_start_date) & (self.historical_data['timestamp'] <= train_end_date)]
        validation_df = self.historical_data[(self.historical_data['timestamp'] >= validation_start_date) & (self.historical_data['timestamp'] <= validation_end_date)]
        test_df = self.historical_data[(self.historical_data['timestamp'] >= test_start_date) & (self.historical_data['timestamp'] <= test_end_date)]

        if self.verbose:
            print(f"Training period: {train_start_date} - {train_end_date}")
            print(f"Validation period: {validation_start_date} - {validation_end_date}")
            print(f"Testing period: {test_start_date} - {test_end_date}")

        return train_df, validation_df, test_df
    

    
    def _test(self, df: pd.DataFrame, n_assets: int):
        for month in list(df["timestamp"].unique()):
            curr_df = df[df["timestamp"] == month].sort_values("performance", ascending=False)
            gt_set = set(curr_df.iloc[:n_assets, :]["symbol"])

            predictions = self.picker.predict(curr_df)

            data = {
                "symbol": curr_df["symbol"].reset_index(drop=True),
                "predicted_return": pd.Series(predictions)
            }

            predictions_final = pd.concat(data, axis=1)

            pred_set = set(predictions_final.sort_values("predicted_return", ascending=False).iloc[:n_assets, :]["symbol"])

            allocations = self.allocator.fit(
                self.past_performance[
                    (self.past_performance.index >= (month - pd.DateOffset(months=self.allocation_period)))
                    &
                    (self.past_performance.index <= month)
                ][list(pred_set)]
            )


            final_result = curr_df[curr_df["symbol"].isin(pred_set)][["timestamp", "symbol", "performance"]]
            final_result['allocations'] = final_result['symbol'].map(allocations)

            avg_return_pred = (final_result['allocations'] * final_result['performance']).sum()
            avg_return_bm = self.benchmark[self.benchmark["timestamp"] == month].iloc[0]["performance"]

            self.test_results["timestamp"].append(month)
            self.test_results["chosen"].append(avg_return_pred)
            self.test_results["benchmark"].append(avg_return_bm)
            self.test_results["correct_selections"].append(len(pred_set.intersection(gt_set)))
            self.test_results["allocations"].append(allocations)

            if self.verbose:
                print(f"Month: {month}, Correct selections: {len(pred_set.intersection(gt_set))}, APR: {avg_return_pred*100:.2f}, ABM: {avg_return_bm*100:.2f}")




    def run(self, n_assets = 50, fit=True) -> pd.DataFrame:
        self.test_results = {
            "correct_selections": [],
            "chosen": [],
            "benchmark": [],
            "timestamp": [],
            "allocations": []
        }
        if self.verbose:
            print(f"Experiment with {self.window} window, {n_assets} assets, {self.picker.tag} choice algo, and {self.allocator.tag} allocator:")

        for i in trange(self.num_periods):
            train_df, val_df, test_df = self._make_dataset(i)
            if fit:
                self.picker.fit(train_df.dropna(), val_df.dropna())
            self._test(test_df.dropna(), n_assets=n_assets)

        return self.test_results

