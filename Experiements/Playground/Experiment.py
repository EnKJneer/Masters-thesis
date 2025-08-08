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

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 1 # Bei RF mit festem random state nicht sinvoll

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataclass = copy.copy(dataSet)
    
    model = mnn.RNN(learning_rate= 0.04834201195017264, n_hidden_size= 94, n_hidden_layers= 1,
                    activation= 'Sigmoid', optimizer_type= 'quasi_newton')

    # Run the experiment
    hexp.run_experiment([dataclass], models=[model],
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name=dataSet.name)