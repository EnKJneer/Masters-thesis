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
import Helper.handling_experiment as hexp
import Models.model_random_forest as mrf
import Models.model_neural_net as mnn
import Models.model_physical as mphy

if __name__ == '__main__':

    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataSet = hdata.DataClass_ST_Plate_Notch

    dataclass = copy.copy(dataSet)

    dataClasses = [dataclass]
    for dataclass in dataClasses:
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values
        dataclass.add_padding = True

    model_rnn = mnn.RNN_noTime(learning_rate=0.1, n_hidden_size=71, n_hidden_layers=1,
                        activation='ELU', optimizer_type='quasi_newton')

    model_nn = mnn.Net(learning_rate= 0.049938580610113505, n_hidden_size= 35, n_hidden_layers= 4,
                               activation= 'ReLU', optimizer_type= 'quasi_newton')

    models = [model_rnn]#model_rnn

    peneltys = [0.01, 1, 100] # 0.1, 10

    for idx, penelty in enumerate(peneltys):
        model = mnn.PiNNThermo(learning_rate=0.1, n_hidden_size=71, n_hidden_layers=1,
                        activation='ELU', optimizer_type= 'quasi_newton', name = f"PiNN_{penelty}", penalty_weight=penelty)
        models.append(model)

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name='PiNN_Thermo')
