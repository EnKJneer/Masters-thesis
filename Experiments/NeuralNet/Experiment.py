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
from datetime import datetime

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataclass1 = hdata.Combined_PlateNotch_OldData
    dataclass2 = hdata.Combined_Plate_TrainVal_OldData

    dataClasses = [dataclass1, dataclass2] #, hdata.Combined_Plate_TrainVal_CONTDEV
    #dataClasses = [hdata.I40_OldData_noAir, hdata.CMX_OldData_noAir]
    for dataclass in dataClasses:
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values
        #dataclass.add_sign_hold = True
        dataclass.use_filter = False

    #model_simple = mphys.NaiveModelSimple()
    model = mnn.get_reference()
    models = [model]

    # Run the experiment
    hexp.run_experiment(dataClasses, use_nn_reference=False, use_rf_reference=True, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        window_size=window_size, past_values=past_values, future_values=future_values, n_drop_values=25,
                        plot_types=['heatmap', 'prediction_overview'])


