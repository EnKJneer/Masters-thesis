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
import Models.model_random_forest as mrf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
def hyperparameter_optimization_ml(folder_path, X_train, X_val, y_train, y_val):
    study_name_nn = "Hyperparameter_Neural_Net_"
    default_parameter_nn = {
        'activation': 'ReLU',
        'window_size': window_size,
        'past_values': past_values,
        'future_values': future_values,
    }
    num_db_files_nn = sum(file.endswith('.db') for file in os.listdir(folder_path))

    while num_db_files_nn < 5:
        search_space_nn = {
            'learning_rate': (0.5e-3, 8e-2),
            'n_hidden_size': (5, 128),
            'n_hidden_layers': (0, 12),
        }
        objective_nn = hyperopt.Objective(
            search_space=search_space_nn,
            model=mnn.Net,
            data=[X_train, X_val, y_train["curr_x"], y_val["curr_x"]],
            n_epochs=NUMBEROFEPOCHS,
            pruning=True
        )
        hyperopt.optimize(objective_nn, folder_path, study_name_nn, n_trials=NUMBEROFTRIALS)
        hyperopt.WriteAsDefault(folder_path, default_parameter_nn)
        num_db_files_nn = sum(file.endswith('.db') for file in os.listdir(folder_path))

    model_params = hyperopt.GetOptimalParameter(folder_path, plot=False)
    print("Neural Network Hyperparameters:", model_params)
    return model_params


if __name__ == "__main__":
    """ Constants """
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 10
    NUMBEROFTRIALS = 250

    window_size = 1
    past_values = 0
    future_values = 0

    #Combined_Gear,Combined_KL
    dataClass_1 = hdata.Combined_Plate_TrainVal
    dataClass_1.window_size = window_size
    dataClass_1.past_values = past_values
    dataClass_1.future_values = future_values

    X_train, X_val, X_test, y_train, y_val, y_test = dataClass_1.load_data()

    folder_path = '..\\..\\Models\\Hyperparameter\\NeuralNet_Plate_Train_Val'
    model_params = hyperparameter_optimization_ml(folder_path, X_train, X_val, y_train, y_val)

    model = mnn.Net(name='Net_optimized', **model_params)

    dataSets_list = [dataClass_1]

    experiment_results = hexp.run_experiment(dataSets_list, True, False, [model],
                        NUMBEROFEPOCHS, NUMBEROFMODELS, past_values, future_values,n_drop_values=25,
                        plot_types=['heatmap', 'heatmap_std', 'prediction_overview'])

    # Zugriff auf Ergebnisse
    print(f"Experiment gespeichert in: {experiment_results['results_dir']}")
    print(f"Anzahl Plots erstellt: {sum(len(paths) for paths in experiment_results['plot_paths'].values())}")