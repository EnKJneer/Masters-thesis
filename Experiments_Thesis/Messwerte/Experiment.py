import copy

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
import Helper.handling_experiment as hexp
import Helper.handling_data as hdata
import Models.model_base as mb
import Models.model_physical as mphys
from Models.model_random_forest import RandomForestModel
from Models.model_neural_net import RNN

if __name__ == '__main__':

    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataclass1 = copy.copy(hdata.DataClass_ST_Plate_Notch_Mes)
    dataclass2 = copy.copy(hdata.DataClass_ST_Plate_Notch)

    dataClasses = [dataclass2, dataclass1]
    for dataclass in dataClasses:
        dataclass.add_padding = True
        dataclass.add_sign_hold = True

    model_rf = RandomForestModel(n_estimators=384, max_depth=435, min_samples_split=4,
                                     min_samples_leaf=2)

    model_rnn = RNN(learning_rate=0.1, n_hidden_size=71, n_hidden_layers=1,
                        activation='ELU', optimizer_type='quasi_newton')


    models = [model_rf]

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'model_heatmap', 'prediction_overview'], experiment_name='Messwerte_RF')
