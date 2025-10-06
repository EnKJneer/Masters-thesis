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
#import Models.model_mixture_of_experts as mmix
from datetime import datetime

def start_experiment():
    """ Constants """
    NUMBEROFTRIALS = 216
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 10

    optimization_samplers = ["TPESampler", "GridSampler"] #

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataSet.add_padding = True

    dataclass = copy.copy(dataSet)

    search_space = {
        "n_hidden_size": (1, 100),
        "n_hidden_layers": (1, 5),
        "learning_rate": (0.001, 0.1),
        "activation": ["ReLU", "Sigmoid", "Tanh", "ELU"],
        "optimizer_type": ["adam", "quasi_newton"]
    }

    model = mnn.PiRNN(learning_rate= 0.1, n_hidden_size= 71, n_hidden_layers= 1,
                      activation= 'ELU', optimizer_type= 'quasi_newton')

    print(f'Anzahl an trials: {NUMBEROFTRIALS}')

    # Run the experiment
    hexp.run_experiment_with_hyperparameteroptimization([dataclass], [model], [search_space],optimization_samplers = optimization_samplers,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, NUMBEROFTRIALS=NUMBEROFTRIALS,
                        plot_types=['heatmap', 'prediction_overview', 'model_heatmap'], experiment_name='Hyperopt_'+model.name)

if __name__ == "__main__":
    start_experiment()