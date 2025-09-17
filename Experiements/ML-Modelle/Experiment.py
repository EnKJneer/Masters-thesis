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

import copy
import datetime
import json
import os
import numpy as np
import pandas as pd
import torch
from numpy.f2py.auxfuncs import throw_error
from scipy.optimize import curve_fit
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
from Experiements.ExpertModels.Experts_3 import Experts_3
#import Models.model_mixture_of_experts as mmix
from datetime import datetime

def start_experiment_for(model_str = 'NN'):
    """ Constants """
    NUMBEROFTRIALS = 512
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 10

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataclass = copy.copy(dataSet)

    optimization_samplers = ["TPESampler", "RandomSampler", "GridSampler"]

    # JSON-Datei laden
    serach_spaces = hyperopt.load_search_spaces('..\\Hyperparameter.json')

    if model_str == 'RF':
        #Random Forest
        search_space = serach_spaces[model_str]
        model = mrf.RandomForestModel()

    elif model_str == 'NN' or model_str == 'LSTM' or model_str == 'RNN':
        search_space = serach_spaces['NN']
        if model_str == 'NN':
            model = mnn.Net()
        elif model_str == 'LSTM':
            model = mnn.LSTM()
            dataclass.add_padding = True
        elif model_str == 'RNN':
            model = mnn.RNN()
            dataclass.add_padding = True

    else:
        throw_error('string is not a valid model')

    print(f'Anzahl an trials: {NUMBEROFTRIALS}')
    # Run the experiment
    hexp.run_experiment_with_hyperparameteroptimization([dataclass], [model], [search_space],optimization_samplers = optimization_samplers,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, NUMBEROFTRIALS=NUMBEROFTRIALS,
                        plot_types=['heatmap', 'prediction_overview', 'model_heatmap'], experiment_name=model.name)

if __name__ == "__main__":
    #start_experiment_for('RF')
    start_experiment_for('NN')
    start_experiment_for('LSTM')
    start_experiment_for('RNN')