import numpy as np
import pandas as pd
import torch
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn
import Models.model_physical as mphys
import Models.model_random_forest as mrf
import Models.model_hyprid as mhy

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 100
    NUMBEROFMODELS = 10

    window_size = 10
    past_values = 2
    future_values = 2

    dataSets = [hdata.Combined_Plate]
    model_erd = mphys.PhysicalModelErd(0.01, 0.01, 0.01, 0.01, 0.01, learning_rate=1)
    model_net = mhy.HybridModelResidual()
    model_rf = mhy.HybridModelResidual(net_model=mrf.get_reference_model())
    models = [model_erd, model_net, model_rf]

    # Run the experiment
    hexp.run_experiment(dataSets, use_nn_reference=True, use_rf_reference=True, models=models, NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, window_size=window_size, past_values=past_values, future_values=future_values)