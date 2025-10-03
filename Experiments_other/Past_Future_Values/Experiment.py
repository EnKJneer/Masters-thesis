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

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 2
    future_values = 2

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataclass1 = copy.copy(dataSet)
    dataclass1.name = 'ZeroValues'
    dataclass1.past_values = 0
    dataclass1.future_values = 0

    dataclass2 = copy.copy(dataSet)
    dataclass2.name = 'Values5'
    dataclass2.past_values = 5
    dataclass2.future_values = 5

    dataclass3 = copy.copy(dataSet)
    dataclass3.name = 'Values10'
    dataclass3.past_values = 10
    dataclass3.future_values = 10

    dataClasses = [dataclass1, dataclass2, dataclass3]
    for dataclass in dataClasses:
        dataclass.window_size = window_size

    model = mrf.get_reference()
    models = [model]

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name=dataSet.name)


