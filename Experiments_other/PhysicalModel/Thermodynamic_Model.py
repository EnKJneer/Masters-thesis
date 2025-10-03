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
    NUMBEROFMODELS = 10  # Bei RF mit festem random state nicht sinvoll

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataclass = copy.copy(dataSet)

    model = mphy.ThermodynamicModel()

    # Run the experiment
    hexp.run_experiment([dataclass], models=[model],
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name='Thermodynamic')