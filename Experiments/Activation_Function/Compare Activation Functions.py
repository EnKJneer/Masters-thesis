import glob
import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Models.model_neural_net as mnn
import Helper.handling_experiment as hexp



if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 10

    window_size = 10
    past_values = 2
    future_values = 2

    dataSets = hdata.dataSets_list_Plate
    # Load data with gear method --> Needed to get input size
    X_train, X_val, X_test, y_train, y_val, y_test = hdata.load_data(dataSets[0], past_values=past_values,
                                                                                       future_values=future_values,
                                                                                       window_size=window_size)

    input_size = X_train.shape[1]
    # Define models to test
    input_size = X_train.shape[1]
    model_nn_ELU = mnn.Net(input_size, 1, input_size, 1, activation=nn.ELU, name='NN_ELU')
    model_nn_PReLU = mnn.Net(input_size, 1, input_size, 1, activation=nn.PReLU, name='NN_PReLU')
    model_nn_Tanh = mnn.Net(input_size, 1, input_size, 1, activation=nn.Tanh, name='NN_Tanh')
    models = [model_nn_ELU, model_nn_PReLU, model_nn_Tanh]

    # Run the experiment
    hexp.run_experiment(dataSets, use_nn_reference=True, use_rf_reference=False, models=models, NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, window_size=window_size, past_values=past_values, future_values=future_values)
