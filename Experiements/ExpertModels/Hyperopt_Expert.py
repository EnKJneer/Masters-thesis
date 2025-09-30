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
import Models.model_physical as mphys
from Experts_2 import Experts_2


def start_experiment():
    """ Constants """
    NUMBEROFTRIALS = 512
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 10

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataclass = copy.copy(dataSet)

    optimization_samplers = ["TPESampler"]

    # JSON-Datei laden
    serach_spaces = hyperopt.load_search_spaces('..\\Hyperparameter.json')


    #Random Forest
    model = mrf.RandomForestModel()

    model_rnn = mnn.RNN(learning_rate=0.09216483876701392, n_hidden_size=104, n_hidden_layers=1,
                        activation='ReLU', optimizer_type='quasi_newton')


    model_phys = mphys.FrictionModel()

    search_space = {
      "n_hidden_size": [13, 130],
      "n_hidden_layers": [1, 5],
      "learning_rate": [0.1, 1],
      "activation": ["ReLU", "Sigmoid", "Tanh", "ELU"],
      "optimizer_type": ["quasi_newton"]
    }

    model = Experts_2()
    model.expert1 = copy.deepcopy(model_phys)  # copy.deepcopy(model_phys)
    model.expert2 = copy.deepcopy(model_rnn)  # mphys.LuGreModelSciPy()

    dataclass.add_padding = True


    print(f'Anzahl an trials: {NUMBEROFTRIALS}')
    # Run the experiment
    hexp.run_experiment_with_hyperparameteroptimization([dataclass], [model], [search_space],optimization_samplers = optimization_samplers,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, NUMBEROFTRIALS=NUMBEROFTRIALS,
                        plot_types=['heatmap', 'prediction_overview', 'model_heatmap'], experiment_name='Hyperopt_Experts')

if __name__ == "__main__":
    start_experiment()