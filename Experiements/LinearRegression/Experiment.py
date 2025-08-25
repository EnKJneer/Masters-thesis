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
import Models.model_physical as mphys


if __name__ == '__main__':

    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataSet.header =["v_x","a_x", "f_x_sim"]
    dataclass1 = copy.copy(dataSet)

    # dataclass2 = hdata.Combined_Plate_TrainVal
    dataClasses = [dataclass1]
    for dataclass in dataClasses:
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values
        dataclass.add_padding = True

    model = mphys.LinearModel()
    models = [model]

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name=dataSet.name)
