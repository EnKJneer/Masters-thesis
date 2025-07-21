import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import optuna
from scipy.optimize import minimize
from collections import deque

import torch
from scipy.optimize import curve_fit
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn
import Models.model_physical as mphys
import Models.model_random_forest as mrf
import Models.model_mixture_of_experts as mmix
import Models.JAX_Version.model_neural_net as jmnn
from datetime import datetime

class EnhancedRandomForestModel:
    def __init__(self, n_estimators=100, max_features=None, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None,
                 name="Enhanced_Random_Forest", epsilon=1,
                 theta_1_init=1.0, theta_2_init=1.0, fifo_length=5):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = None
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.name = name
        self.device = "cpu"
        self.epsilon = epsilon
        self.theta_1_init = {col: theta_1_init for col in ['v_x_1_current']} #, 'v_y_1_current', 'v_z_1_current', 'v_sp_1_current'
        self.theta_2_init = {col: theta_2_init for col in ['v_x_1_current']} # , 'v_y_1_current', 'v_z_1_current', 'v_sp_1_current'
        self.theta_1 = self.theta_1_init.copy()
        self.theta_2 = self.theta_2_init.copy()
        self.fifo_length = fifo_length
        self.v_columns = ['v_x_1_current'] # , 'v_y_1_current', 'v_z_1_current', 'v_sp_1_current'
        self.z_method = 'combined'

    def _compute_z_feature_sign_hold(self, df, v_columns, theta_1_dict, theta_2_dict):
        z_features = {}
        for col in v_columns:
            v = df[col].values
            z = np.zeros_like(v)
            theta_1 = theta_1_dict.get(col, self.theta_1_init[col])
            theta_2 = theta_2_dict.get(col, self.theta_2_init[col])
            for i in range(1, len(v)):
                z[i] = z[i-1] + (theta_1 * v[i] - theta_2 * z[i-1])
                z[i] = np.clip(z[i], v.min()*10, v.max()*10)
            z_features[col] = z
        return z_features

    def _compute_z_feature_pure_sign_hold(self, df, v_columns):
        z_features = {}
        for col in v_columns:
            v = df[col].values
            z = np.zeros(len(v))
            h = deque([0] * self.fifo_length, maxlen=self.fifo_length)
            for i in range(len(v)):
                if abs(v[i]) > self.epsilon:
                    h.append(v[i])
                if i >= self.fifo_length - 1:
                    z[i] = np.sign(sum(h))
            z_features[col] = z
        return z_features

    def _add_z_feature(self, X, theta_1_dict=None, theta_2_dict=None):
        if theta_1_dict is None:
            theta_1_dict = self.theta_1
        if theta_2_dict is None:
            theta_2_dict = self.theta_2
        X_enhanced = X.copy()
        if self.v_columns is None:
            possible_columns = [col for col in X.columns if any(v in col.lower() for v in ['v_x', 'v_y', 'v_z', 'v_sp', 'vx', 'vy', 'vz', 'vsp'])]
            if possible_columns:
                self.v_columns = possible_columns
            else:
                raise ValueError("No velocity columns found. Please specify v_columns manually.")

        if self.z_method == 'combined':
            z_values = self._compute_z_feature_sign_hold(X_enhanced, self.v_columns, theta_1_dict, theta_2_dict)
        elif self.z_method == 'sign_hold':
            z_values = self._compute_z_feature_pure_sign_hold(X_enhanced, self.v_columns)
        else:
            raise ValueError("Method must be 'sign_hold' or 'combined'")

        for col, z in z_values.items():
            X_enhanced[f'z_feature_{col}'] = z
        return X_enhanced

    def _objective_function(self, theta_params, X_train, y_train, X_val, y_val):
        if self.z_method == 'sign_hold':
            theta_1_dict = {col: 0.1 for col in self.v_columns}
            theta_2_dict = {col: 10 for col in self.v_columns}
        else:
            theta_1_dict = {col: theta_params[i] for i, col in enumerate(self.v_columns)}
            theta_2_dict = {col: theta_params[i + len(self.v_columns)] for i, col in enumerate(self.v_columns)}

        X_train_enhanced = self._add_z_feature(X_train, theta_1_dict, theta_2_dict)
        X_val_enhanced = self._add_z_feature(X_val, theta_1_dict, theta_2_dict)

        temp_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            max_depth=self.max_depth if hasattr(self, 'max_depth') else None,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        temp_model.fit(X_train_enhanced, y_train.squeeze())
        y_val_pred = temp_model.predict(X_val_enhanced)
        return mean_squared_error(y_val, y_val_pred)

    def optimize_theta_parameters(self, X_train, y_train, X_val, y_val,
                                  v_columns=None, optimization_method='scipy', n_trials=100):
        if v_columns is not None:
            self.v_columns = v_columns
        if isinstance(X_train, list):
            X_train = pd.concat(X_train, ignore_index=True)
            y_train = pd.concat(y_train, ignore_index=True)
        if isinstance(X_val, list):
            X_val = pd.concat(X_val, ignore_index=True)
            y_val = pd.concat(y_val, ignore_index=True)

        if self.z_method == 'sign_hold':
            print("Using pure sign_hold method - no theta optimization needed.")
            return {
                'theta_1': self.theta_1,
                'theta_2': self.theta_2,
                'method': self.z_method,
                'best_validation_error': self._objective_function([1.0] * 2 * len(self.v_columns), X_train, y_train, X_val, y_val)
            }

        if optimization_method == 'scipy':
            initial_params = [self.theta_1_init[col] for col in self.v_columns] + [self.theta_2_init[col] for col in self.v_columns]
            bounds = [(0.1, 10.0)] * 2 * len(self.v_columns)
            result = minimize(
                self._objective_function,
                x0=initial_params,
                args=(X_train, y_train, X_val, y_val),
                method='L-BFGS-B',
                bounds=bounds
            )
            for i, col in enumerate(self.v_columns):
                self.theta_1[col] = result.x[i]
                self.theta_2[col] = result.x[i + len(self.v_columns)]

            return {
                'theta_1': self.theta_1,
                'theta_2': self.theta_2,
                'method': self.z_method,
                'optimization_result': result,
                'best_validation_error': result.fun
            }

        elif optimization_method == 'optuna':
            def optuna_objective(trial):
                theta_1_dict = {col: trial.suggest_float(f'theta_1_{col}', 0.1, 10.0) for col in self.v_columns}
                theta_2_dict = {col: trial.suggest_float(f'theta_2_{col}', 0.1, 10.0) for col in self.v_columns}
                theta_params = [theta_1_dict[col] for col in self.v_columns] + [theta_2_dict[col] for col in self.v_columns]
                return self._objective_function(theta_params, X_train, y_train, X_val, y_val)

            study = optuna.create_study(direction='minimize')
            study.optimize(optuna_objective, n_trials=n_trials)

            for col in self.v_columns:
                self.theta_1[col] = study.best_params[f'theta_1_{col}']
                self.theta_2[col] = study.best_params[f'theta_2_{col}']

            return {
                'theta_1': self.theta_1,
                'theta_2': self.theta_2,
                'method': self.z_method,
                'study': study,
                'best_validation_error': study.best_value
            }
        else:
            raise ValueError("Optimization method must be 'scipy' or 'optuna'")

    def criterion(self, y_target, y_pred):
        return mean_squared_error(y_target, y_pred)

    def predict(self, X):
        X_enhanced = self._add_z_feature(X)
        return self.model.predict(X_enhanced)

    def train_model(self, X_train, y_train, X_val, y_val, v_columns=None, z_method='combined', optimize_theta=True,
                    optimization_method='scipy', n_trials=100, n_epochs=1, trial=None, draw_loss=False, n_outlier=12, patience=10):
        if v_columns is not None:
            self.v_columns = v_columns
        if isinstance(X_train, list):
            X_train = pd.concat(X_train, ignore_index=True)
            y_train = pd.concat(y_train, ignore_index=True)
        if isinstance(X_val, list):
            X_val = pd.concat(X_val, ignore_index=True)
            y_val = pd.concat(y_val, ignore_index=True)

        self.z_method = z_method
        best_val_error = float('inf')

        if optimize_theta and self.z_method != 'sign_hold':
            print(f"Optimizing theta parameters for {self.z_method} method using {optimization_method}...")
            optimization_result = self.optimize_theta_parameters(
                X_train, y_train, X_val, y_val,
                v_columns=self.v_columns,
                optimization_method=optimization_method,
                n_trials=n_trials
            )
            print(f"Optimized theta_1: {self.theta_1}, theta_2: {self.theta_2}")
            print(f"Best validation error from optimization: {optimization_result['best_validation_error']:.4f}")
        elif self.z_method == 'sign_hold':
            print("Using pure sign_hold method - no theta optimization needed.")

        X_train_enhanced = self._add_z_feature(X_train)
        X_val_enhanced = self._add_z_feature(X_val)

        self.model.fit(X_train_enhanced, y_train.squeeze())
        y_val_pred = self.model.predict(X_val_enhanced)
        val_error = self.criterion(y_val, y_val_pred)

        if trial:
            trial.report(val_error, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        best_val_error = val_error

        if draw_loss:
            plt.figure(figsize=(10, 6))
            plt.plot([0], [val_error], 'bo-', label='Validation Loss')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title(f'Training Progress - {self.z_method} method')
            plt.legend()
            plt.grid(True)
            plt.show()

        print(f'{self.name} ({self.z_method}): Final Val Error: {val_error:.4f}')
        return best_val_error

    def test_model(self, X, y_target, criterion_test=None):
        if criterion_test is None:
            criterion_test = self.criterion
        y_pred = self.predict(X)
        loss = criterion_test(y_target, y_pred)
        return loss, y_pred

    def compare_methods(self, X_train, y_train, X_val, y_val, v_columns=None):
        methods = ['combined', 'sign_hold']
        results = {}
        for method in methods:
            print(f"\n--- Testing {method} method ---")
            temp_model = EnhancedRandomForestModel(
                n_estimators=self.n_estimators,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                epsilon=self.epsilon,
                theta_1_init=self.theta_1_init['v_x'],
                theta_2_init=self.theta_2_init['v_x'],
                fifo_length=self.fifo_length,
                name=f"{self.name}_{method}"
            )
            val_error = temp_model.train_model(
                X_train, y_train, X_val, y_val,
                v_columns=v_columns,
                z_method=method,
                optimize_theta=(method != 'sign_hold'),
                optimization_method='scipy'
            )
            results[method] = {
                'validation_error': val_error,
                'theta_1': temp_model.theta_1,
                'theta_2': temp_model.theta_2,
                'model': temp_model
            }

        print("\n--- Method Comparison ---")
        for method, result in results.items():
            print(f"{method}: Val Error = {result['validation_error']:.4f}, "
                  f"θ₁ = {result['theta_1']}, θ₂ = {result['theta_2']}")
        return results

    def get_documentation(self):
        documentation = {
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_features": self.max_features,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "epsilon": self.epsilon,
                "theta_1": self.theta_1,
                "theta_2": self.theta_2,
                "fifo_length": self.fifo_length
            },
            "feature_engineering": {
                "methods": {
                    "sign_hold": "z = sign(sum(FiFo_buffer)) with FiFo updated when abs(v) > epsilon",
                    "combined": "z = sign(sum(FiFo_buffer)) * magnitude * decay_factor"
                },
                "v_columns": self.v_columns,
                "current_method": self.z_method
            }
        }
        return documentation
if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1
    NUMBEROFMODELS = 1

    window_size = 1
    past_values = 0
    future_values = 0

    dataclass1 = hdata.Combined_PlateNotch_TrainVal_OldData
    #dataclass2 = hdata.Combined_Plate_TrainVal
    dataClasses = [dataclass1]
    for dataclass in dataClasses:
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values
        dataclass.add_sign_hold = False
        dataclass.use_filter = False

    # Selbe Hyperparameter wie referenz
    model = EnhancedRandomForestModel(n_estimators=10, max_features = None, min_samples_split = 2, min_samples_leaf = 1)
    model2 = mrf.RandomForestModel(n_estimators=10, max_features = None, min_samples_split = 2, min_samples_leaf = 1)
    model2.name = 'test'
    models = [model]

    # Run the experiment
    hexp.run_experiment(dataClasses, use_nn_reference=False, use_rf_reference=True, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        window_size=window_size, past_values=past_values, future_values=future_values,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name='FiFo')