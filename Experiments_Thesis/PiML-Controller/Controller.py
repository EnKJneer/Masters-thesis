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

    integrate = False

    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    controller_input = ["cont_dev_x", "cont_dev_y", "cont_dev_z"]

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataSet.header = hdata.HEADER_x
    dataclass1 = copy.deepcopy(dataSet)
    dataclass1.name = '1_no_cont_dev'

    dataclass2 = copy.copy(dataSet)
    dataclass2.name = '2_with_cont_dev'
    for col in controller_input:
        dataclass2.header.append(col)

    dataclass3 = copy.copy(dataSet)
    dataclass3.name = '3_int_cont_dev'
    for col in controller_input:
        dataclass3.header.append(col)
    dataclass3.columns_to_integrate = controller_input

    dataClasses = [dataclass1, dataclass2, dataclass3]
    for dataclass in dataClasses:
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values
        dataclass.add_padding = True

    model_rf = mrf.RandomForestModel(n_estimators=384, max_depth=435, min_samples_split=4,
                                     min_samples_leaf=2)

    model_rnn = mnn.RNN(learning_rate= 0.1, n_hidden_size= 71, n_hidden_layers= 1,
                    activation= 'ELU', optimizer_type= 'quasi_newton')

    models = [model_rnn, model_rf]

    use_rf = True
    if use_rf:
        models = [model_rf]
        name = '_RF'
    else:
        models = [model_rnn]
        name = '_RNN'

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name='Controller'+name)
