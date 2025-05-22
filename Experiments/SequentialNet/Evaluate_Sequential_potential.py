import glob
import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 2

    window_size = 10
    past_values = 2
    future_values = 2

    dataSets = hdata.Combined_Plate

    # Define models to test]
    model_nn = mnn.get_reference_net()
    model_rnn = mnn.PartialRnn(None, 1, None, 1)
    models = [model_rnn]

    # Run the experiment
    hexp.run_experiment([dataSets], use_nn_reference=True, use_rf_reference=False, models=models, NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, window_size=window_size, past_values=past_values, future_values=future_values)
