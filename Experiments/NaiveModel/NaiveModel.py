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
from datetime import datetime

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    #Combined_Gear,Combined_KL
    dataClass_1 = hdata.Combined_Plate_TrainVal
    dataClass_1.window_size = window_size
    dataClass_1.past_values = past_values
    dataClass_1.future_values = future_values
    dataClass_1.target_channels = ['curr_y']
    dataSets_list = [dataClass_1]

    #model_simple = mphys.NaiveModelSimple()
    model = mphys.NaiveModelSigmoid(target_channel = dataClass_1.target_channels[0])
    model_2 = mphys.NaiveModel(target_channel = dataClass_1.target_channels[0])
    model_erd = mphys.PhysicalModelErdSingleAxis(target_channel = dataClass_1.target_channels[0])
    models = [model_2, model] #model_erd

    # Run the experiment
    hexp.run_experiment(dataSets_list, use_nn_reference=True, use_rf_reference=False, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        window_size=window_size, past_values=past_values, future_values=future_values,
                        plot_types=['heatmap', 'prediction_overview', 'geometry_mae'])


