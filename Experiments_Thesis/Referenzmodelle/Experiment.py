import copy

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
import Helper.handling_experiment as hexp
import Helper.handling_data as hdata
import Models.model_base as mb
import Models.model_physical as mphys
import Models.model_random_forest as mrf
import Models.model_neural_net as mnn


if __name__ == '__main__':

    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataSet = hdata.DataClass_ST_Plate_Notch_Reference

    dataclass1 = copy.copy(dataSet)
    axis = 'z'
    dataclass1.target_channels = [f'curr_{axis}']

    dataClasses = [dataclass1]
    for dataclass in dataClasses:
        dataclass.add_padding = True

    model_rf = mrf.RandomForestModel(n_estimators=481, max_depth=133, min_samples_split=6,
                                     min_samples_leaf=5)

    model_rnn = mnn.RNN(learning_rate=0.1, n_hidden_size=71, n_hidden_layers=1,
                        activation='ELU', optimizer_type='quasi_newton')
    model = model_rnn
    models = [model]

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'model_heatmap', 'prediction_overview'], experiment_name=model.name+'_'+axis)
