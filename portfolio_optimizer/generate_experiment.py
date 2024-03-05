from portfolio_optimizer.stock_choice import MLChoice, NaiveChoice, EMAChoice, SMAChoice
from portfolio_optimizer.weight_allocators import SimpleAllocator, HierarchicalRiskParity
from portfolio_optimizer import PortfolioOptimizer
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def _save_ml_experiments(save_pth: str, optimizer: PortfolioOptimizer) -> None:
    filename_model = f'{save_pth}/models/{optimizer.picker.tag}-{optimizer.train_period}-{optimizer.window}.sav'
    pickle.dump(optimizer.picker.best_model, open(filename_model, 'wb'))

    #create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': optimizer.picker.best_model.feature_names_in_,
        'importance': optimizer.picker.best_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    #plot variable importances of the model
    filename_fig = f'{save_pth}/figures/{optimizer.picker.tag}-{optimizer.train_period}-{optimizer.window}.png'
    plt.title('Variable Importances', fontsize=16)
    sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
    plt.savefig(filename_fig)

def run_experiments(
        choice_df: pd.DataFrame,
        alloc_df: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        save_pth: str,
        times = [3, 6, 12], 
        train_periods = [36, 60], 
        n_assests = [10, 20, 50, 100],
        allocation_periods = [24, 36],
        windows = ['fixed', 'rolling', 'expanding'],
        models = ['xgboost', 'random_forest']
    ) -> dict:
    simple_pickers = [NaiveChoice()]
    ml_pickers: list[MLChoice] = []

    for time in times:
        simple_pickers.extend([SMAChoice(time), EMAChoice(time)])
    for model in models:
        ml_pickers.append(MLChoice(model))

    if os.path.exists(f"{save_pth}/results/cached_results.pickle"):
        with open(f"{save_pth}/results/cached_results.pickle", 'rb') as f:
            all_results = pickle.load(f)
    else:
        all_results = {}

    optimizer = PortfolioOptimizer(
        choice_df,
        SimpleAllocator(),
        NaiveChoice(),
        benchmark=benchmark_df,
        train_period=36,
        past_performance=alloc_df
    )

    for picker in tqdm(simple_pickers, desc=f"Pickers", position=0):
        if f"correct_assets{n_assests[-1]}_{picker.tag}_HRP-{allocation_periods[-1]}" in all_results:
            print("Skipping...")
            continue
        optimizer.picker = picker
        for i, num_asset in enumerate(tqdm(n_assests, desc=f"Assets", position=1)):
            optimizer.allocator = SimpleAllocator()

            results = optimizer.run(n_assets=num_asset)
            if not(all_results):
                all_results = {key: results[key] for key in ["timestamp", "benchmark"]}
            all_results[f"returns_assets{num_asset}_{optimizer.picker.tag}_{optimizer.allocator.tag}"] = results["chosen"]
            all_results[f"correct_assets{num_asset}_{optimizer.picker.tag}_{optimizer.allocator.tag}"] = results["correct_selections"]

            for alloc_period in allocation_periods:
                optimizer.allocator = HierarchicalRiskParity()
                optimizer.allocation_period = alloc_period

                results = optimizer.run(n_assets=num_asset)
                all_results[f"returns_assets{num_asset}_{optimizer.picker.tag}_{optimizer.allocator.tag}-{alloc_period}"] = results["chosen"]
                all_results[f"correct_assets{num_asset}_{optimizer.picker.tag}_{optimizer.allocator.tag}-{alloc_period}"] = results["correct_selections"]
        
        pickle.dump(all_results, open(f"{save_pth}/results/cached_results.pickle", 'wb'))

    for window in tqdm(windows, desc=f"Window", position=2):
        optimizer = PortfolioOptimizer(
            choice_df,
            SimpleAllocator(),
            NaiveChoice(),
            benchmark=benchmark_df,
            train_period=36,
            past_performance=alloc_df,
            window=window, 
            verbose=True
        )
        for train_period in tqdm(train_periods, desc=f"Train period", position=3):
            optimizer.train_period = train_period
            for picker in tqdm(ml_pickers, desc=f"Pickers", position=4):
                optimizer.picker = picker
                if window == "fixed":
                    for i, num_asset in enumerate(tqdm(n_assests, desc=f"Assets", position=5)):
                        optimizer.allocator = SimpleAllocator()

                        if i==0:
                            results = optimizer.run(num_asset, fit=True)
                        else:
                            results = optimizer.run(num_asset, fit=False)    

                        all_results[f"returns_assets{num_asset}_{optimizer.picker.tag}-{train_period}-{window}_{optimizer.allocator.tag}"] = results["chosen"]
                        all_results[f"correct_assets{num_asset}_{optimizer.picker.tag}-{train_period}-{window}_{optimizer.allocator.tag}"] = results["correct_selections"]
                        all_results[f"alloc_assets{num_asset}_{optimizer.picker.tag}-{train_period}-{window}_{optimizer.allocator.tag}"] = results["allocations"]

                        for alloc_period in allocation_periods:
                            optimizer.allocator = HierarchicalRiskParity()
                            optimizer.allocation_period = alloc_period

                            results = optimizer.run(num_asset, fit=False)    
                            all_results[f"returns_assets{num_asset}_{optimizer.picker.tag}-{train_period}-{window}_{optimizer.allocator.tag}-{alloc_period}"] = results["chosen"]
                            all_results[f"correct_assets{num_asset}_{optimizer.picker.tag}-{train_period}-{window}_{optimizer.allocator.tag}-{alloc_period}"] = results["correct_selections"]
                            all_results[f"alloc_assets{num_asset}_{optimizer.picker.tag}-{train_period}-{window}_{optimizer.allocator.tag}-{alloc_period}"] = results["allocations"]
                else:
                    if f"correct_assets50_{picker.tag}-{train_period}-{window}_HRP-{allocation_periods[-1]}" in all_results:
                        print("Skipping...")
                        continue
                    optimizer.allocator = SimpleAllocator()
                    results = optimizer.run(fit=True)

                    all_results[f"returns_assets50_{optimizer.picker.tag}-{train_period}-{window}_{optimizer.allocator.tag}"] = results["chosen"][1:]
                    all_results[f"correct_assets50_{optimizer.picker.tag}-{train_period}-{window}_{optimizer.allocator.tag}"] = results["correct_selections"][1:]
                    all_results[f"alloc_assets50_{optimizer.picker.tag}-{train_period}-{window}_{optimizer.allocator.tag}"] = results["allocations"][1:]
                    pickle.dump(all_results, open(f"{save_pth}/results/cached_results.pickle", 'wb'))
                    for alloc_period in allocation_periods:
                        optimizer.allocator = HierarchicalRiskParity()
                        optimizer.allocation_period = alloc_period

                        results = optimizer.run(fit=True)    
                        all_results[f"returns_assets50_{optimizer.picker.tag}-{train_period}-{window}_{optimizer.allocator.tag}-{alloc_period}"] = results["chosen"][1:]
                        all_results[f"correct_assets50_{optimizer.picker.tag}-{train_period}-{window}_{optimizer.allocator.tag}-{alloc_period}"] = results["correct_selections"][1:]
                        all_results[f"alloc_assets50_{optimizer.picker.tag}-{train_period}-{window}_{optimizer.allocator.tag}-{alloc_period}"] = results["allocations"][1:]

                _save_ml_experiments(save_pth, optimizer)
                pickle.dump(all_results, open(f"{save_pth}/results/cached_results.pickle", 'wb'))

    final_results = pd.DataFrame.from_dict(all_results)
    final_results.to_csv(f"{save_pth}/results/final_results.csv")

    return final_results