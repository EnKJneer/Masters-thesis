import copy
import datetime
import json
import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from torch import nn
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
    f_s = 0.28
    a_f = 0.001
    F_c = 0.56
    sigma_2 = 0.001
    a_b = 0.005

    model_simulation = mphys.FrictionModel(f_s=f_s, a_x=a_f, a_sp= 0.1*a_f, b=0, f_c=F_c, sigma_2=sigma_2, a_b=a_b)

    def simulate_y(x):
        y = model_simulation.predict(x)
        #y_fy = 0.5 * a_f * x['f_y_sim_1_current'].values
        y_fz = 0.5 * a_f * x['f_z_sim_1_current'].values
        y_mrr = -0.5 * a_f * x['materialremoved_sim_1_current'].values
        y = y  + y_fz + y_mrr #+ y_fy

        # Erstelle eine zufällige Variation für jedes Element in y
        noise = np.random.uniform(-0.05, 0.05, size=y.shape)
        y = y * (1 + noise)
        return pd.DataFrame(-y, columns=['curr_x'])

    model = mphys.FrictionModel()
    model_rf = mrf.RandomForestModel()

    dataClasses = [hdata.Combined_Plate_TrainVal]
    results = []

    for dataClass in dataClasses:
        dataClass.window_size = window_size
        dataClass.past_values = past_values
        dataClass.future_values = future_values
        dataClass.add_sign_hold = True
        dataClass.target_channels = ['curr_x']

        X_train, X_val, X_test, _, _, y_test_original = dataClass.load_data()
        y_train = simulate_y(X_train)
        y_val = simulate_y(X_val)

        model.train_model(X_train, y_train, X_val, y_val)
        model_rf.train_model(X_train, y_train, X_val, y_val)

        n_drop = 20
        for idx, x in enumerate(X_test):
            name = dataClass.testing_data_paths[idx]
            print(name)
            y_test = simulate_y(x)
            mae, y_test_pred = model.test_model(x, y_test)
            print(f'mae friction: {mae:.3f}')
            mae_rf, y_test_pred_rf = model_rf.test_model(x, y_test)
            print(f'mae rf: {mae_rf:.3f}')

            # Sammeln der MAE-Werte
            results.append({
                'DataSet': dataClass.name,
                'DataPath': os.path.basename(name),
                'Model': 'FrictionModel',
                'MAE': np.mean(np.abs(y_test.values.squeeze()[:-n_drop] - y_test_pred[:-n_drop]))
            })
            results.append({
                'DataSet': dataClass.name,
                'DataPath': os.path.basename(name),
                'Model': 'RandomForestModel',
                'MAE': np.mean(np.abs(y_test.values.squeeze()[:-n_drop] - y_test_pred_rf[:-n_drop]))
            })

            # Plotte die Daten
            plt.figure(figsize=(10, 6))
            plt.plot(y_test.values[:-n_drop], color='black', label='y_test')
            plt.plot(y_test_pred[:-n_drop], label='y_test_pred')
            plt.plot(y_test_pred_rf[:-n_drop], label='y_test_pred_rf')
            plt.plot(y_test_original[idx].values[:-n_drop], 'r--', label='y_test_original', alpha=0.5)
            plt.xlabel('Index')
            plt.ylabel('Wert')
            plt.title(name)
            plt.legend()
            plt.show()

    # Erstellen der Heatmap
    df_results = pd.DataFrame(results)
    df_results['Dataset_Path'] = df_results['DataSet'] + '_' + df_results['DataPath'].str.replace('.csv', '')
    pivot_df = df_results.pivot_table(values='MAE', index='Dataset_Path', columns='Model', aggfunc='mean')

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_df.index) * 0.5)))
    im = ax.imshow(pivot_df.values, cmap='RdYlBu_r', aspect='auto')

    # Achsenbeschriftungen
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticklabels(pivot_df.index)

    # Rotiere die x-Achsenbeschriftungen
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Werte in die Zellen schreiben
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            value = pivot_df.iloc[i, j]
            if not np.isnan(value):
                text = ax.text(j, i, f'{value:.3f}', ha="center", va="center",
                               color="white" if value > pivot_df.values.mean() else "black")

    ax.set_title("MAE Heatmap: Models vs Dataset_DataPath")
    fig.colorbar(im, ax=ax, label='MAE')
    plt.tight_layout()
    plt.savefig('heatmap_mae.png')
    plt.show()
