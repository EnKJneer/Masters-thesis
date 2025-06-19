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
import Models.JAX_Version.model_physical as jmphys
import Models.model_random_forest as mrf
import Models.model_mixture_of_experts as mmix
from datetime import datetime

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 250
    NUMBEROFMODELS = 1

    window_size = 1
    past_values = 0
    future_values = 0

    dataClasses = [hdata.PPhys, hdata.Combined_PPhys_SF, hdata.Combined_PPhys_Depth, hdata.Combined_PPhys]
    #for dataClass in dataClasses:
    #Combined_Gear,Combined_KL
    dataClass_1 = hdata.Combined_PPhys
    dataClass_1.window_size = window_size
    dataClass_1.past_values = past_values
    dataClass_1.future_values = future_values
    #dataClass_1.keep_separate = True
    dataClass_1.target_channels = ['curr_x']

    dataSets_list = [dataClass_1]

    #model_simple = mphys.NaiveModelSimple()
    model = mphys.FrictionModel()
    models = [model] # ,

    # Run the experiment
    hexp.run_experiment(dataSets_list, use_nn_reference=False, use_rf_reference=False, use_phys_reference=True, models = models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        window_size=window_size, past_values=past_values, future_values=future_values,
                        plot_types=['heatmap', 'prediction_overview'])
