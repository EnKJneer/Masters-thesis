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
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataclass1 = copy.deepcopy(dataSet)
    dataclass1.name = 'ohne cont_dev'
    dataclass2 = copy.copy(dataSet)
    dataclass2.name = 'mit cont_dev'
    dataclass2.header = ["v_sp", "v_x", "v_y", "v_z",
                         "a_x", "a_y", "a_z", "a_sp",
                         "f_x_sim", "f_y_sim", "f_z_sim", "f_sp_sim",
                         "cont_dev_x", "cont_dev_y", "cont_dev_z",
                         "materialremoved_sim"]
    dataclass2.columns_to_integrate = ["cont_dev_x", "cont_dev_y", "cont_dev_z"]
    # dataclass2 = hdata.Combined_Plate_TrainVal
    dataClasses = [dataclass1, dataclass2]
    for dataclass in dataClasses:
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values
        dataclass.add_padding = True

    model_rf = mrf.RandomForestModel(n_estimators= 52,max_features= 500, min_samples_split= 67,
                    min_samples_leaf= 4)
    model_rnn = mnn.RNN(learning_rate= 0.04834201195017264, n_hidden_size= 94, n_hidden_layers= 1,
                    activation= 'Sigmoid', optimizer_type= 'quasi_newton')
    models = [model_rnn, model_rf]

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name=dataSet.name)
