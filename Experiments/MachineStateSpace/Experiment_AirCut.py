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
import Models.JAX_Version.model_neural_net as jmnn
import Models.JAX_Version.model_physical as jmphys
from datetime import datetime

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 1

    window_size = 1
    past_values = 0
    future_values = 0

    dataclass2 = hdata.AirCutNo_Plate
    dataclass2.target_channels = ['curr_x']

    dataClasses = [dataclass2]
    for dataclass in dataClasses:
        dataclass.only_aircut = True
        dataclass.remove_bias = False
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values
        dataclass.keep_separate = True
        #dataclass.add_sign_hold = True

    #model_simple = mphys.NaiveModelSimple()
    model = jmphys.ODEModelJAX()
    model_2 = jmphys.LinearModelJAX()
    models = [model] #, model_2

    # Run the experiment
    hexp.run_experiment(dataClasses, use_nn_reference=False, use_rf_reference=True, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        window_size=window_size, past_values=past_values, future_values=future_values, n_drop_values=1,
                        plot_types=['heatmap', 'prediction_overview'])