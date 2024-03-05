import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from portfolio_optimizer.stock_choice.NaiveChoice import NaiveChoice
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from itertools import product

class MLChoice(NaiveChoice):
    def __init__(self, model: str, validation: str = 'mse') -> None:
        self.validation = validation

        if model == 'xgboost':
            self.models = self._make_xgboost()
        elif model == "random_forest":
            self.models = self._make_random_forest()
        else:
            raise NotImplementedError(f"Model {model} is not implemented yet!")
        
        self.tag = f"MLChoice-{model}" 


    def _make_xgboost(self) -> list:
        param_grid = {
            'n_estimators': [100],
            'max_depth': [6],
            'learning_rate': [0.1]
        }

        if self.validation == 'mae':
            objective = 'reg:pseudohubererror'
        else:
            objective = 'reg:squarederror'

        self.param_combinations = list(product(*param_grid.values()))

        models = []

        for params in self.param_combinations:
            model = XGBRegressor(
                objective=objective,
                n_estimators=params[0],
                max_depth=params[1],
                learning_rate=params[2]
            )
            models.append(model)

        return models
    

    def _make_random_forest(self) -> list:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [8, 10, 12],
            'min_samples_leaf': [20, 50, 80]
        }

        self.param_combinations = list(product(*param_grid.values()))

        models = []

        for params in self.param_combinations:
            model = RandomForestRegressor(
                n_estimators=params[0],
                max_depth=params[1],
                min_samples_leaf=params[2]
            )
            models.append(model)

        return models


    def predict(self, df: pd.DataFrame):
        X_test = df.drop(['performance', 'symbol', 'timestamp', 'label'], axis=1)
        return self.best_model.predict(X_test)
    
    def score(self, df: pd.DataFrame):
        X_test = df.drop(['performance', 'symbol', 'timestamp', 'label'], axis=1)
        y_test = df['performance']
        r_squared_oos = 1 - (
            mean_squared_error(y_test, self.best_model.predict(X_test))
            /
            mean_squared_error(y_test, np.zeros(y_test.shape))
        )
        return (r_squared_oos, 
                mean_squared_error(y_test, self.best_model.predict(X_test)),
                mean_absolute_error(y_test, self.best_model.predict(X_test)))

    
    def fit(self, df: pd.DataFrame, df_val: pd.DataFrame, n_assets: int = 50):
        X_train = df.drop(['performance', 'symbol', 'timestamp', 'label'], axis=1)
        y_train = df['performance']

        X_val = df_val.drop(['performance', 'symbol', 'timestamp', 'label'], axis=1)
        y_val = df_val['performance']

        self.best_model = None
        self.best_mean_return = -float('inf')

        for i, model in enumerate(pbar := tqdm(self.models, desc=f"Hyperprameter optimization", leave=False)):
            model.fit(X_train, y_train)
            if self.validation == 'mse':
                y_pred = model.predict(X_val)
                val_metric = 1/mean_squared_error(y_val, y_pred)
            elif self.validation == 'mae':
                y_pred = model.predict(X_val)
                val_metric = 1/mean_absolute_error(y_val, y_pred)
            else:
                raise RuntimeError(f'Validation {self.validation} not yet implemented')
            
            if val_metric > self.best_mean_return:
                pbar.set_postfix_str(f"Best params: {self.param_combinations[i]}, return: {val_metric}")
                self.best_model = model
                self.best_mean_return = val_metric

        predicitons_train = self.best_model.predict(X_train)

        self.best_model = self.best_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))