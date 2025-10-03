import copy
import datetime
import json
import os

import numpy as np
import optuna
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy.f2py.auxfuncs import throw_error
from torch import nn
from scipy.optimize import curve_fit
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import Helper.handling_data as hdata
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_base as mb
import Models.model_neural_net as mnn
import Models.model_physical as mphys
import Models.model_random_forest as mrf
import Models.model_mixture_of_experts as mmix
from datetime import datetime


class MachineAndProcessLinearCombined(mb.BaseModel):
    def __init__(self, *args, name='MachineAndProcessModel_Linear',
                 n_estimators=100, max_features = 1, max_depth =None, min_samples_split = 2, min_samples_leaf = 1, random_state=None,
                 **kwargs):
        super(MachineAndProcessLinearCombined, self).__init__(*args, name, **kwargs)
        self.model_process = mrf.RandomForestRegressor(n_estimators=n_estimators, max_features = max_features,
                                                       max_depth = max_depth, min_samples_split = min_samples_split,
                                                       min_samples_leaf = min_samples_leaf, random_state=random_state,
                                                       n_jobs = -1)
        self.model_machine = LinearRegression()
        self.input_machine = ['a_x_1_current', 'a_y_1_current', 'a_z_1_current', 'a_sp_1_current']

    def criterion(self, y_target, y_pred):
        """
        Compute the Mean Squared Error (MSE) loss between the target and predicted values.

        Parameters
        ----------
        y_target : array-like
            The target values.
        y_pred : array-like
            The predicted values.

        Returns
        -------
        float
            The computed MSE loss.
        """
        return mean_squared_error(y_target, y_pred)

    def predict(self, X):
        """
        Make predictions based on the input data using the Random Forest model.

        Parameters
        ----------
        X : array-like
            The input data.

        Returns
        -------
        numpy.ndarray
            The predicted values.
        """
        y_machine = self.model_machine.predict(X[self.input_machine]).squeeze()
        y_process = self.model_process.predict(X)
        return y_process + y_machine

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=1, trial=None, draw_loss=False, n_outlier=12, patience_stop=10):
        """
        Train the Random Forest model using the training data and validate it using the validation data.

        Parameters
        ----------
        X_train : array-like
            The training input data.
        y_train : array-like
            The training target values.
        X_val : array-like
            The validation input data.
        y_val : array-like
            The validation target values.
        n_epochs : int, optional
            The number of epochs for training. Default is 1 (since Random Forest is not iterative).
        trial : optuna.trial.Trial, optional
            An Optuna trial object used for pruning based on intermediate validation errors.
        draw_loss : bool, optional
            If True, plots training and validation loss after each epoch. Default is False.
        n_outlier: int, optional
            Number of std used to filter out outliers. Default is 12.
        Returns
        -------
        best_val_error : float
            The best validation error achieved during training.
        """
        epsilon = 1e-3
        best_val_error = float('inf')
        if type(X_train) is list:
            X_train = pd.concat(X_train, ignore_index=True)
            y_train = pd.concat(y_train, ignore_index=True)
        if type(X_val) is list:
            X_val = pd.concat(X_val, ignore_index=True)
            y_val = pd.concat(y_val, ignore_index=True)

        # Daten trennen
        x_aircut = X_train[abs(X_train['materialremoved_sim_1_current']) < epsilon]
        y_aircut = y_train[abs(X_train['materialremoved_sim_1_current']) < epsilon]

        x_process = X_train[abs(X_train['materialremoved_sim_1_current']) >= epsilon]
        y_process = y_train[abs(X_train['materialremoved_sim_1_current']) >= epsilon]

        # Training loop only useful for hyperparameteroptimization
        if trial is None:
            n_epochs= 1
        for epoch in range(n_epochs):
            x_lin = x_aircut[self.input_machine]
            # Maschinen Modell trainieren
            self.model_machine.fit(x_lin, y_aircut)
            y_process_no_machine = y_process - self.model_machine.predict(x_process[self.input_machine])

            self.model_process.fit(x_process, y_process_no_machine.squeeze())

            y_val_pred = self.predict(X_val)
            val_error = self.criterion(y_val, y_val_pred)

            # Report intermediate values to the pruner
            if trial:
                trial.report(val_error, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Update the best validation error
            if val_error < best_val_error:
                best_val_error = val_error

            if draw_loss:
                plt.plot(epoch, val_error, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

            print(
                f'{self.name}: Epoch {epoch + 1}/{n_epochs},  Val Error: {val_error:.4f}')

        return best_val_error
    def reset_hyperparameter(self):
        throw_error('Not implemented')

    def test_model(self, X, y_target, criterion_test=None):
        """
        Test the model using the test data and compute the loss.

        Parameters
        ----------
        X : array-like
            The test input data.
        y_target : array-like
            The test target values.

        Returns
        -------
        tuple
            A tuple containing the loss and the predicted values.
        """
        if criterion_test is None:
            criterion_test = self.criterion
        y_pred = self.predict(X)
        loss = criterion_test(y_target, y_pred)
        return loss, y_pred

    def get_documentation(self):
        documentation = {"hyperparameters": {
            "n_estimators": self.model_process.n_estimators,
            "max_features": self.model_process.max_features,
            "min_samples_split": self.model_process.min_samples_split,
            "min_samples_leaf": self.model_process.min_samples_leaf,
            "input_machine": self.input_machine,
        }}
        return documentation

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 10

    window_size = 10
    past_values = 0
    future_values = 0

    dataclass2 = hdata.DataClass_ST_Plate_Notch
    dataclass2.target_channels = ['curr_x']

    dataClasses = [dataclass2]
    for dataclass in dataClasses:
        #dataclass.only_aircut = True
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values

    #model_simple = mphys.NaiveModelSimple()
    model = MachineAndProcessLinearCombined(n_estimators= 100, max_depth= 100, max_features = None,
                                     min_samples_split= 2, min_samples_leaf= 4)

    models = [model]

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['model_heatmap', 'prediction_overview'], experiment_name='Addition')