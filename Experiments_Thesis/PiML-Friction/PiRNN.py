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
import Models.model_physical as mphy
if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 10

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataSet.add_padding = True

    dataclass = copy.deepcopy(dataSet)

    model_rnn = copy.deepcopy(mnn.RNN(learning_rate= 0.1, n_hidden_size= 71, n_hidden_layers= 1,
                    activation= 'ELU', optimizer_type= 'quasi_newton'))

    model = mnn.PiRNN(learning_rate= 0.1, n_hidden_size= 71, n_hidden_layers= 1,
                      activation= 'ELU', optimizer_type= 'quasi_newton')

    # Run the experiment
    hexp.run_experiment([dataclass], models=[model, model_rnn],
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview', 'model_heatmap'], experiment_name='PiRNN')
