import os

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


def hyperparameter_optimization_PiNN(folder_path, X_train, X_val, y_train, y_val):
    study_name_nn = "Hyperparameter_Neural_Net_"
    default_parameter_nn = {
        'activation': 'ReLU',
        'window_size': window_size,
        'past_values': past_values,
        'future_values': future_values,
    }
    num_db_files_nn = sum(file.endswith('.db') for file in os.listdir(folder_path))

    while num_db_files_nn < 4:
        search_space_nn = {
            'learning_rate': (0.5e-3, 8e-2),
            'n_hidden_size': (5, 128),
            'n_hidden_layers': (0, 12),
            'penalty_weight': (1, 500)
        }
        objective_nn = hyperopt.Objective(
            search_space=search_space_nn,
            model=mnn.PiNN,
            data=[X_train, X_val, y_train["curr_x"], y_val["curr_x"]],
            n_epochs=NUMBEROFEPOCHS,
            pruning=True
        )
        hyperopt.optimize(objective_nn, folder_path, study_name_nn, n_trials=NUMBEROFTRIALS)
        hyperopt.WriteAsDefault(folder_path, default_parameter_nn)
        num_db_files_nn = sum(file.endswith('.db') for file in os.listdir(folder_path))

    model_params = hyperopt.GetOptimalParameter(folder_path, plot=False)
    print("Hyperparameters:", model_params)
    return model_params

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 2

    window_size = 1
    past_values = 0
    future_values = 0

    #Combined_Gear,Combined_KL
    dataClass_1 = hdata.Combined_PK_TrainVal
    dataClass_1.window_size = window_size
    dataClass_1.past_values = past_values
    dataClass_1.future_values = future_values

    dataSets_list = [dataClass_1]

    #X_train, X_val, X_test, y_train, y_val, y_test = dataClass_1.load_data()

    #folder_path = '..\\..\\Models\\Hyperparameter\\PiNN_PK_Train_Val'
    #model_params = hyperparameter_optimization_PiNN(folder_path, X_train, X_val, y_train, y_val)

    #model = mnn.PiNN(name='PiNN_optimized', **model_params)

    #model_erd = mphys.PhysicalModelErd(learning_rate=1)

    model_pinn = mnn.PiNNNaiveLinear(penalty_weight=50)
    model_pinn_naive = mnn.PiNNNaive()

    model_pinn_adaptive = mnn.PiNNAdaptiv(penalty_weight=50)
    #model_pinn_adaptive_newton = mnn.PiNNAdaptiv(name='Pinn_adaptive_newton', c_1=1.415e-05, c_2=-2.42e-06, c_3=-4.79e-06, penalty_weight=50, learning_rate=1, optimizer_type='quasi_newton')

    model_pinn_matrix = mnn.PiNNErdMatrix(theta_init=np.array([[1.52348093e-05, -6.92416393e-07, -3.38071050e-06, -1.44576282e-10, -1.06608064e-11],
                                                               [-3.34966649e-06, -9.13457313e-07,  3.79570906e-06, -1.32213537e-11, -1.06608064e-11],
                                                               [ 2.84573389e-07, -5.36836637e-08,  9.28007239e-07, -1.75489189e-11, -1.06608064e-11],
                                                               [-8.91113905e-10, -7.01598113e-10,  2.98052828e-06, -1.00204480e-11, -1.06608064e-11]]))
    models = [model_pinn_naive, model_pinn, model_pinn_matrix] # model_pinn_adaptive,

    # Run the experiment
    hexp.run_experiment(dataSets_list, use_nn_reference=True, use_rf_reference=False, models=models, NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, window_size=window_size, past_values=past_values, future_values=future_values)