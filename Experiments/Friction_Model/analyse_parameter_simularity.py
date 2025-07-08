import copy
import datetime
import json
import os
import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from torch import nn
import matplotlib.pyplot as plt
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn
import Models.model_physical as mphys
import Models.JAX_Version.model_physical as jmphys
import Models.model_random_forest as mrf
import Models.model_mixture_of_experts as mmix
from datetime import datetime

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1
    NUMBEROFMODELS = 1
    window_size = 1
    past_values = 0
    future_values = 0
    #dataClasses = [hdata.Combined_Plate_TrainVal] #, hdata.Combined_Plate_St_TrainVal
    dataClasses = [hdata.Combined_OldData_noAir]
    for dataClass in dataClasses:
        dataClass.window_size = window_size
        dataClass.past_values = past_values
        dataClass.future_values = future_values
        dataClass.add_sign_hold = True
        dataClass.target_channels = ['curr_x']

    model = mphys.FrictionModel()
    model.target_channel = 'curr_x'
    models = [model]

    results = []

    for i, dataClass in enumerate(dataClasses):
        print(f"\n===== Verarbeitung: {dataClass.name} =====")

        # Daten laden
        X_train, X_val, X_test, y_train, y_val, y_test = dataClass.load_data()
        model.target_channel = dataClass.target_channels[0]

        for i, (x, y) in enumerate(zip(X_test, y_test)):
            path = dataClass.testing_data_paths[i]
            model.train_model(x, y, X_val, y_val, patience=5)
            mae, pred_nn = model.test_model(x, y)

            # Speichern der Parameter und des MAE
            parameters = {
                'dataset': os.path.basename(path),
                'F_s': model.F_s,
                'theta_f': model.theta_f,
                'b': model.b,
                'F_c': model.F_c,
                'sigma_2': model.sigma_2,
                'theta_a': model.theta_a,
                'mae': mae
            }
            results.append(parameters)

            print(f"{model.name}: Test MAE: {mae}")

    # Grafische Darstellung der Parameter
    parameters_df = pd.DataFrame(results)
    parameters_df.set_index('dataset', inplace=False)

    parameters_df.plot(kind='bar', subplots=True, layout=(-1, 3), figsize=(15, 10), legend=False)
    plt.tight_layout()
    # Zentrale Legende hinzufügen
    legend_text = "\n".join([f"{i}: {dataset}" for i, dataset in enumerate(parameters_df['dataset'])])
    plt.figtext(0.5, 0.01, legend_text, ha='center', fontsize=12, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

    plt.show()
