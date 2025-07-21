import copy
import datetime
import json
import os

import numpy as np
import pandas as pd
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import optuna
from scipy.optimize import minimize


class EnhancedRandomForestModel:
    def __init__(self, n_estimators=100, max_features=1, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None,
                 name="Enhanced_Random_Forest", epsilon=2,
                 theta_1_init=1.0, theta_2_init=1.0):
        """
        Initializes an Enhanced Random Forest regressor with feature engineering.

        Parameters
        ----------
        n_estimators : int, optional
            The number of trees in the Random Forest. The default is 100.
        epsilon : float, optional
            Threshold for v_x values. Default is 1e-3.
        theta_1_init : float, optional
            Initial value for theta_1 parameter. Default is 1.0.
        theta_2_init : float, optional
            Initial value for theta_2 parameter. Default is 1.0.
        """
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

        # Save parameters for documentation
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.name = name
        self.device = "cpu"

        # New parameters for feature engineering
        self.epsilon = epsilon
        self.theta_1 = theta_1_init
        self.theta_2 = theta_2_init
        self.theta_1_init = theta_1_init
        self.theta_2_init = theta_2_init

        # For storing v_x column name
        self.v_x_column = None

    def _compute_z_feature(self, df, v_x_column, theta_1, theta_2):
        """
        Compute the z feature based on the given formula.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe containing v_x values
        v_x_column : str
            Name of the v_x column in the dataframe
        theta_1 : float
            Parameter theta_1
        theta_2 : float
            Parameter theta_2

        Returns
        -------
        numpy.ndarray
            Computed z values
        """
        v_x = df[v_x_column].values
        z = np.zeros_like(v_x)

        # Initialize tracking variables
        v_0 = None
        t = 0

        for i in range(len(v_x)):
            if abs(v_x[i]) > self.epsilon:
                # Case 1: abs(v_x) > epsilon
                z[i] = v_x[i]
                v_0 = v_x[i]  # Update v_0 to current v_x
                t = 0  # Reset counter
            else:
                # Case 2: abs(v_x) <= epsilon
                t += 1
                if v_0 is not None:
                    z[i] = v_0*(theta_1 + np.exp(-t * theta_2))
                else:
                    # If no v_0 found yet, use v_x directly
                    z[i] = v_x[i]

        return z

    def _add_z_feature(self, X, theta_1=None, theta_2=None):
        """
        Add the z feature to the input dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            Input dataframe
        theta_1 : float, optional
            Parameter theta_1. If None, uses self.theta_1
        theta_2 : float, optional
            Parameter theta_2. If None, uses self.theta_2

        Returns
        -------
        pandas.DataFrame
            Dataframe with added z feature
        """
        if theta_1 is None:
            theta_1 = self.theta_1
        if theta_2 is None:
            theta_2 = self.theta_2

        X_enhanced = X.copy()

        # Auto-detect v_x column if not set
        if self.v_x_column is None:
            # Try to find column containing 'v_x' or similar
            possible_columns = [col for col in X.columns if 'v_x' in col.lower() or 'vx' in col.lower()]
            if possible_columns:
                self.v_x_column = possible_columns[0]
            else:
                raise ValueError("v_x column not found. Please specify v_x_column manually.")

        # Compute z feature
        z_values = self._compute_z_feature(X_enhanced, self.v_x_column, theta_1, theta_2)
        X_enhanced['z_feature'] = z_values

        return X_enhanced

    def _objective_function(self, theta_params, X_train, y_train, X_val, y_val):
        """
        Objective function for theta optimization.

        Parameters
        ----------
        theta_params : list
            [theta_1, theta_2] parameters
        X_train : pandas.DataFrame
            Training data
        y_train : pandas.Series
            Training targets
        X_val : pandas.DataFrame
            Validation data
        y_val : pandas.Series
            Validation targets

        Returns
        -------
        float
            Validation error
        """
        theta_1, theta_2 = theta_params

        # Add z feature with current theta values
        X_train_enhanced = self._add_z_feature(X_train, theta_1, theta_2)
        X_val_enhanced = self._add_z_feature(X_val, theta_1, theta_2)

        # Train model
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
                                  v_x_column=None, method='scipy', n_trials=100):
        """
        Optimize theta_1 and theta_2 parameters.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training data
        y_train : pandas.Series
            Training targets
        X_val : pandas.DataFrame
            Validation data
        y_val : pandas.Series
            Validation targets
        v_x_column : str, optional
            Name of the v_x column. If None, auto-detect.
        method : str, optional
            Optimization method: 'scipy' or 'optuna'. Default is 'scipy'.
        n_trials : int, optional
            Number of trials for Optuna optimization. Default is 100.

        Returns
        -------
        dict
            Dictionary containing optimized parameters and optimization results
        """
        if v_x_column is not None:
            self.v_x_column = v_x_column

        # Ensure data is in correct format
        if isinstance(X_train, list):
            X_train = pd.concat(X_train, ignore_index=True)
            y_train = pd.concat(y_train, ignore_index=True)
        if isinstance(X_val, list):
            X_val = pd.concat(X_val, ignore_index=True)
            y_val = pd.concat(y_val, ignore_index=True)

        if method == 'scipy':
            # Use scipy optimization
            result = minimize(
                self._objective_function,
                x0=[self.theta_1_init, self.theta_2_init],
                args=(X_train, y_train, X_val, y_val),
                method='L-BFGS-B',
                bounds=[(0.1, 10.0), (0.1, 10.0)]  # Reasonable bounds for theta parameters
            )

            self.theta_1, self.theta_2 = result.x

            return {
                'theta_1': self.theta_1,
                'theta_2': self.theta_2,
                'optimization_result': result,
                'best_validation_error': result.fun
            }

        elif method == 'optuna':
            # Use Optuna optimization
            def optuna_objective(trial):
                theta_1 = trial.suggest_float('theta_1', 0.1, 10.0)
                theta_2 = trial.suggest_float('theta_2', 0.1, 10.0)
                return self._objective_function([theta_1, theta_2], X_train, y_train, X_val, y_val)

            study = optuna.create_study(direction='minimize')
            study.optimize(optuna_objective, n_trials=n_trials)

            self.theta_1 = study.best_params['theta_1']
            self.theta_2 = study.best_params['theta_2']

            return {
                'theta_1': self.theta_1,
                'theta_2': self.theta_2,
                'study': study,
                'best_validation_error': study.best_value
            }

        else:
            raise ValueError("Method must be 'scipy' or 'optuna'")

    def criterion(self, y_target, y_pred):
        """
        Compute the Mean Squared Error (MSE) loss between the target and predicted values.
        """
        return mean_squared_error(y_target, y_pred)

    def predict(self, X):
        """
        Make predictions based on the input data using the enhanced Random Forest model.
        """
        X_enhanced = self._add_z_feature(X)
        return self.model.predict(X_enhanced)

    def train_model(self, X_train, y_train, X_val, y_val, v_x_column=None,
                    optimize_theta=True, optimization_method='scipy', n_trials=100,
                    n_epochs=1, trial=None, draw_loss=False, n_outlier=12, patience=10):
        """
        Train the Enhanced Random Forest model.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training data
        y_train : pandas.Series
            Training targets
        X_val : pandas.DataFrame
            Validation data
        y_val : pandas.Series
            Validation targets
        v_x_column : str, optional
            Name of the v_x column. If None, auto-detect.
        optimize_theta : bool, optional
            Whether to optimize theta parameters. Default is True.
        optimization_method : str, optional
            Method for theta optimization: 'scipy' or 'optuna'. Default is 'scipy'.
        n_trials : int, optional
            Number of trials for Optuna optimization. Default is 100.
        """
        if v_x_column is not None:
            self.v_x_column = v_x_column

        # Ensure data is in correct format
        if isinstance(X_train, list):
            X_train = pd.concat(X_train, ignore_index=True)
            y_train = pd.concat(y_train, ignore_index=True)
        if isinstance(X_val, list):
            X_val = pd.concat(X_val, ignore_index=True)
            y_val = pd.concat(y_val, ignore_index=True)

        best_val_error = float('inf')

        # Optimize theta parameters if requested
        if optimize_theta:
            print(f"Optimizing theta parameters using {optimization_method}...")
            optimization_result = self.optimize_theta_parameters(
                X_train, y_train, X_val, y_val,
                v_x_column=self.v_x_column,
                method=optimization_method,
                n_trials=n_trials
            )
            print(f"Optimized theta_1: {self.theta_1:.4f}, theta_2: {self.theta_2:.4f}")
            print(f"Best validation error from optimization: {optimization_result['best_validation_error']:.4f}")

        # Add z feature with optimized (or initial) theta values
        X_train_enhanced = self._add_z_feature(X_train)
        X_val_enhanced = self._add_z_feature(X_val)

        # Train the model
        self.model.fit(X_train_enhanced, y_train.squeeze())
        y_val_pred = self.model.predict(X_val_enhanced)
        val_error = self.criterion(y_val, y_val_pred)

        # Report to Optuna trial if provided
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
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
            plt.show()

        print(f'{self.name}: Final Val Error: {val_error:.4f}')

        return best_val_error

    def test_model(self, X, y_target, criterion_test=None):
        """
        Test the model using the test data and compute the loss.
        """
        if criterion_test is None:
            criterion_test = self.criterion
        y_pred = self.predict(X)
        loss = criterion_test(y_target, y_pred)
        return loss, y_pred

    def get_documentation(self):
        """
        Get model documentation including hyperparameters and theta values.
        """
        documentation = {
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_features": self.max_features,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "epsilon": self.epsilon,
                "theta_1": self.theta_1,
                "theta_2": self.theta_2
            },
            "feature_engineering": {
                "z_feature_formula": "z = v_x if abs(v_x) > epsilon else v_0 * (theta_1 + exp(-t**theta_2))",
                "v_x_column": self.v_x_column
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

    #model_simple = mphys.NaiveModelSimple()
    model = EnhancedRandomForestModel()
    models = [model]

    # Run the experiment
    hexp.run_experiment(dataClasses, use_nn_reference=False, use_rf_reference=True, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        window_size=window_size, past_values=past_values, future_values=future_values,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name='Notch_vs_Plate')