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


if __name__ == '__main__':

    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 1

    window_size = 1
    past_values = 0
    future_values = 0

    dataSet = hdata.DataClass_ST_Plate_Notch
    #dataSet.header = ["v_x", "a_x", "f_x_sim", "materialremoved_sim"]
    dataclass2 = copy.copy(dataSet)
    #dataclass2.name = 'mit z'
    #dataclass2.add_sign_hold = True
    #dataclass2.target_channels = ['curr_y']
    # dataclass2 = hdata.Combined_Plate_TrainVal
    dataClasses = [dataclass2]
    for dataclass in dataClasses:
        dataclass.future_values = future_values
        dataclass.add_padding = True

    torch.backends.cudnn.enabled = False

    model_rf = mrf.RandomForestModel(n_estimators= 167, min_samples_split= 6,
                    min_samples_leaf= 2)

    model_rnn = mnn.RNN(learning_rate= 0.01, n_hidden_size= 130, n_hidden_layers= 1,
                    activation= 'Tanh', optimizer_type= 'quasi_newton')

    model_nn = mnn.Net(learning_rate= 0.07307859730865025, n_hidden_size= 69, n_hidden_layers= 5,
                    activation= 'ELU', optimizer_type= 'quasi_newton')

    models = [model_rnn]


    # Run the experiment
    hexp.run_experiment_with_shap(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name=dataSet.name,
                                  block_size = 100, stride = 200)
