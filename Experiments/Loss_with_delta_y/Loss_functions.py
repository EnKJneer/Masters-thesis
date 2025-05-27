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
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 10

    window_size = 10
    past_values = 2
    future_values = 2


    def criterion_only_delta(y_target, y_pred):
        criterion = nn.MSELoss()
        dy_target = torch.diff(y_target, dim=0)
        dy_pred = torch.diff(y_pred,dim=0)
        return 100*criterion(dy_target, dy_pred)

    def criterion_with_delta(y_target, y_pred):
        criterion = nn.MSELoss()
        dy_target = torch.diff(y_target, dim=0)
        dy_pred = torch.diff(y_pred,dim=0)
        return 10000*criterion(dy_target, dy_pred) + criterion(y_target, y_pred)

    dataSets = [hdata.Combined_Plate]
    model_dy = mnn.get_reference()
    model_dy.criterion = criterion_only_delta
    model_dy.name = 'Net_dy'
    model_net = mnn.get_reference()
    model_net.criterion = criterion_with_delta
    model_net.name = 'Net_y_dy'
    model_small = mnn.get_reference()
    model_small.n_hidden_size = 5
    model_small.name = 'Net_small'
    models = [model_net, model_dy, model_small]

    # Run the experiment
    hexp.run_experiment(dataSets, use_nn_reference=True, use_rf_reference=False, models=models, NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, window_size=window_size, past_values=past_values, future_values=future_values)