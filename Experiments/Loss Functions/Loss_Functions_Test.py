import numpy as np
import pandas as pd
import torch
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn
import Models.model_physical as mphys
import Models.model_random_forest as mrf

from sklearn.metrics import mean_squared_error, root_mean_squared_error

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 2

    window_size = 1
    past_values = 2
    future_values = 2

    def criterion(y_target, y_pred):
        criterion = nn.MSELoss()
        return criterion(y_target.squeeze(), y_pred.squeeze())

    def criterion_rf(y_target, y_pred):
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
        return root_mean_squared_error(y_target, y_pred)

    dataSets = [hdata.Combined_Plate]
    model_rmse = mnn.get_reference()
    model_rmse.criterion = criterion
    model_rmse.name = "NN_mse"

    model_rf_rmse = mrf.get_reference()
    model_rf_rmse.criterion = criterion_rf
    model_rf_rmse.name = "RF_rmse"

    models = [model_rmse, model_rf_rmse]

    # Run the experiment
    hexp.run_experiment(dataSets, use_nn_reference=True, use_rf_reference=True, models=models, NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, window_size=window_size, past_values=past_values, future_values=future_values)