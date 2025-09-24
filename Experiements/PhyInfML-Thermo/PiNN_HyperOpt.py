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

def start_experiment_for():
    """ Constants """
    NUMBEROFTRIALS = 256
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 10

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataclass = copy.copy(dataSet)

    optimization_samplers = ["TPESampler"] #, "RandomSampler", "GridSampler"

    search_space ={"n_hidden_size": [13, 130],
                   "n_hidden_layers": [1, 5],
                   "activation": ["ReLU", "Sigmoid", "Tanh", "ELU"],
                   "penalty_weight": [0.1, 100]}

    model = mnn.PiRNNThermo(learning_rate= 0.1, n_hidden_size= 104, n_hidden_layers= 1,
                    activation= 'ReLU', optimizer_type= 'quasi_newton', name = f"PiNN_Thermo")
    dataclass.add_padding = True


    print(f'Anzahl an trials: {NUMBEROFTRIALS}')
    # Run the experiment
    hexp.run_experiment_with_hyperparameteroptimization([dataclass], [model], [search_space],optimization_samplers = optimization_samplers,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, NUMBEROFTRIALS=NUMBEROFTRIALS,
                        plot_types=['heatmap', 'prediction_overview', 'model_heatmap'], experiment_name=model.name)

if __name__ == "__main__":
    start_experiment_for()
