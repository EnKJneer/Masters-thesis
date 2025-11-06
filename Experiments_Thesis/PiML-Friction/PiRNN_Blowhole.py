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

    dataSet = hdata.DataClass_ST_Plate_Notch

    dataSet.testing_data_paths = ['DMC60H_AL2007T4_Gear_Blowhole_3.csv', 'DMC60H_AL2007T4_Plate_Blowhole_3.csv',
                      'DMC60H_S235JR_Gear_Blowhole_3.csv', 'DMC60H_S235JR_Plate_Blowhole_3.csv']

    dataclass1 = copy.copy(dataSet)
    axis = 'y'
    dataclass1.target_channels = [f'curr_{axis}']
    dataClasses = [dataclass1]
    for dataclass in dataClasses:
        dataclass.add_padding = True

    model_rnn = copy.deepcopy(mnn.RNN(learning_rate= 0.1, n_hidden_size= 71, n_hidden_layers= 1,
                    activation= 'ELU', optimizer_type= 'quasi_newton'))

    model = mnn.LuGre_PiRNN(learning_rate= 0.1, n_hidden_size= 71, n_hidden_layers= 1,
                            activation= 'ELU', optimizer_type= 'quasi_newton')
    model.target_channel = 'curr_y'

    # Run the experiment
    hexp.run_experiment(dataClasses, models=[model], #, model_rnn
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview', 'model_heatmap'], experiment_name=f'PiRNN_Blowhole_{axis}')