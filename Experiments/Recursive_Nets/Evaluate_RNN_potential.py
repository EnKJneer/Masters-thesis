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
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn


""" Functions """

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 1

    window_size = 1
    past_values = 0
    future_values = 0

    dataClasses = [hdata.Combined_Plate_TrainVal] # , hdata.Combined_Plate_St_TrainVal
    for dataClass in dataClasses:
        dataClass.window_size = window_size
        dataClass.past_values = past_values
        dataClass.future_values = future_values
        dataClass.add_sign_hold = True
        dataClass.target_channels = ['curr_x']

    n_hidden_size = 2
    input_size = None
    model_rnn = mnn.RNN(input_size, 1, input_size, n_hidden_size)
    model_lstm = mnn.LSTM(input_size, 1, input_size, n_hidden_size)
    model_gru = mnn.GRU(input_size, 1, input_size, n_hidden_size)
    model_partial_rnn = mnn.PartialRnn(input_size, 1, input_size, n_hidden_size)
    model_partial_gru = mnn.PartialGRU(input_size, 1, input_size, n_hidden_size)

    models = [model_rnn, model_lstm, model_gru] #, model_partial_rnn, model_partial_gru

    # Run the experiment
    hexp.run_experiment(dataClasses, use_nn_reference=True, use_rf_reference=False, models=models, NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, window_size=window_size, past_values=past_values, future_values=future_values)