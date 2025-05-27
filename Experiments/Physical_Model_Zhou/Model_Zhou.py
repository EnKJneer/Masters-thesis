import glob
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
import Models.model_physical as mphy
import Helper.handling_experiment as hexp
from scipy.optimize import least_squares
"""
Problem:    Nur zwei verschiedene Materialien, nicht genug um Material Abhängigkeit zu lernen  
            Daher verwendung von Modell aus:
                A new energy consumption model suitable for processing multiple materials in end milling
                Zhou (2021) 
                DOI: 10.1007/s00170-021-07078-3
            Allerdings Gleichung 17 n hat negativen exponenten kleiner 1 -> n darf nur positiv und ungleich 0 sein.
            in der realität nicht gegebn
"""
# Modellklasse
class Model:
    def __init__(self, theta):
        self.theta = theta
        self.name = 'Zhou'

    def __call__(self, input):
        x = get_x(input[0])
        MRR = x[:, 0]
        n = x[:, 1]
        v_x = x[:, 2]
        v_y = x[:, 3]
        H = input[1]  # Beispielwert für H, anpassen falls nötig
        output_mat = 0.0012 * (np.abs(n)**0.0360) * (H**0.1773) * MRR #0.0012 * (n**-0.0360) * (H**0.1773) * MRR
        output_spin = self.theta[0] * n
        output_feed = self.theta[1] * np.sqrt(v_x**2 + v_y**2)
        output = output_mat + output_spin + output_feed + self.theta[2]
        return output
    def get_documentation(self):
        documentation = {"hyperparameters": {
            "optimizer": 'LevMarq',
        }}
        return documentation

# Residuenfunktion
def residuals(theta, x_data, y_data):
    model = Model(theta)
    predictions = model(x_data)
    return (predictions - y_data.values.T).flatten()   # Sicherstellen, dass ein 1D-Array zurückgegeben wird


def get_x(df):
    prefix_current = '_1_current'
    return df[['materialremoved_sim' + prefix_current, 'v_sp' + prefix_current, 'v_x' + prefix_current,
               'v_y' + prefix_current]].values

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 50
    NUMBEROFMODELS = 2

    window_size = 10
    past_values = 2
    future_values = 2

    dataSets = hdata.Combined_Plate

    model_Zhou = mphy.ModelZhou()
    initial_theta = np.array([0.1, 0.1, 0.1])
    models = [Model(initial_theta)]

    # Run the experiment
    use_nn_reference = True
    use_rf_reference = False
    batched_data = False
    n_drop_values = 10
    if type(dataSets) is not list:
        dataSets = [dataSets]

    # Create directory for results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = os.path.join("Results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Define the meta information structure
    meta_information = {
        "DataSets": [],
        "Models": [],
        "Data_Preprocessing": {
            "window_size": window_size,
            "past_values": past_values,
            "future_values": future_values,
            "batched_data": batched_data,
            "n_drop_values": n_drop_values,
        }
    }

    # Define reference models
    reference_models = []
    if use_nn_reference:
        model_nn = mnn.get_reference()
        reference_models.append(model_nn)
    if use_rf_reference:
        model_rf = mrf.get_reference_model()
        reference_models.append(model_rf)

    # Add reference models to meta information
    for model in reference_models:
        model_info = {
            model.name+"_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                **model.get_documentation()
            },
        }
        meta_information["Models"].append(model_info)

    # Add models to meta information
    for model in models:
        model_info = {
            model.name+"_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                **model.get_documentation()
            },
        }
        meta_information["Models"].append(model_info)

    # Add datasets to meta information
    for data_params in dataSets:
        data_info = {
            "name": data_params.name,
            "folder": data_params.folder,
            "training_validation_data": data_params.training_validation_datas,
            "testing_data_paths": data_params.testing_data_paths,
            "target_channels": data_params.target_channels,
            "percentage_used": data_params.percentage_used
        }
        meta_information["DataSets"].append(data_info)

    # Save the meta information to a JSON file
    documentation = meta_information

    """ Check if data_params.testing_data_paths is the same """

    """ Prediction """
    results = []
    for i, data in enumerate(dataSets):
        print(f"\n===== Verarbeitung: {data.name} =====")
        # Daten laden
        X_train, X_val, X_test, y_train, y_val, y_test = hdata.load_data(
            data, past_values, future_values, window_size, keep_separate=batched_data)
        if isinstance(X_test, list):
            df_list_results = [pd.DataFrame() for x in range(len(X_test))]
            header_list = [[] for x in range(len(X_test))]
        else:
            df_list_results = [pd.DataFrame()]
            header_list = []

        # Train and test reference models
        for model in reference_models:
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            for _ in range(NUMBEROFMODELS):
                model.train_model(X_train, y_train, X_val, y_val, NUMBEROFEPOCHS)
                if isinstance(X_test, list):
                    for i, (x, y) in enumerate(zip(X_test, y_test)):
                        _, pred_nn = model.test_model(x, y)
                        nn_preds[i].append(pred_nn.flatten())
                else:
                    _, pred_nn = model.test_model(X_test, y_test)
                    nn_preds.append(pred_nn.flatten())

            # Fehlerberechnung
            for i, path in enumerate(data.testing_data_paths):
                name = model.name + "_" + path.replace('.csv', '')

                mse_nn, std_nn = hdata.calculate_mse_and_std(nn_preds[i], y_test[i].values if isinstance(y_test[i],
                                                                                                   pd.DataFrame) else y_test[i],
                                                             n_drop_values)
                # Ergebnisse speichern
                df_list_results[i][name] = np.mean(nn_preds[i], axis=0)
                results.extend([
                    [data.name + "_" + path.replace('.csv', ''), model.name, mse_nn, std_nn],
                ])
                header_list[i].append(name)

        X_train, X_val, X_test, y_train, y_val, y_test = hdata.load_data_with_material_check(
            data, past_values, future_values, window_size, keep_separate=batched_data)

        # Train and test models
        for model in models:
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            # Anfangswerte für theta
            initial_theta = np.array([0.1, 0.1, 0.1])

            # Optimierung mit Levenberg-Marquardt
            result = least_squares(residuals, initial_theta, args=(X_train, y_train), method='lm')

            # Optimierte Parameter
            optimized_theta = result.x
            print("Optimierte Parameter:", optimized_theta)
            model = Model(optimized_theta)
            for x, h in zip(X_test[0], X_test[1]):
                pred_nn = model((x, h))
                nn_preds.append(pred_nn.flatten())

            # Fehlerberechnung
            for i, path in enumerate(data.testing_data_paths):
                name = model.name + "_" + path.replace('.csv', '')

                mse_nn, std_nn = hdata.calculate_mse_and_std(nn_preds, y_test, n_drop_values)
                # Ergebnisse speichern
                df_list_results[i][name] = np.mean(nn_preds[i], axis=0)
                results.extend([
                    [data.name + "_" + path.replace('.csv', ''), model.name, mse_nn, std_nn],
                ])
                header_list[i].append(name)

        """ Dokumentation """
        df_list_data, names = hdata.get_test_data_as_pd(data, past_values=past_values, future_values=future_values,
                                               window_size=window_size)
        data_dir = os.path.join(results_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        file_paths = [os.path.join(data_dir, f'{data.name}_{name}.csv') for name in names]
        hdata.save_data(df_list_data, file_paths)

        # Save Results in csv
        for y, df, header, path in zip(y_test, df_list_results, header_list, data.testing_data_paths):
            name = path.replace('.csv', '')
            df['y_ground_truth'] = y["curr_x"] if isinstance(y, pd.DataFrame) else y

            # Plot speichern
            plots_dir = os.path.join(results_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            plt.figure(figsize=(12, 6))
            plt.plot(df['y_ground_truth'], label='Ground Truth', color='black', linewidth=2)

            for col in header:
                label = col.replace('_'+name,'')
                plt.plot(df[col], linestyle='--', label=f'{label}', alpha=0.6)

            plt.title(f'{data.name}_{name}: Modellvergleich')
            plt.xlabel('Zeit')
            plt.ylabel('Strom in A')
            plt.legend(loc='best')
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(plots_dir, f'{data.name}_{name}_comparison.png')
            plt.savefig(plot_path)
            plt.close()

        hdata.add_pd_to_csv(df_list_data, file_paths, header_list)

    # Export als CSV
    # Ordner anlegen
    os.makedirs("Data", exist_ok=True)

    df = pd.DataFrame(results, columns=["DataSet", "Model", "MSE", "StdDev"])
    models = df['Model'].unique()
    n_models = len(models)
    datasets = sorted(df['DataSet'].unique())

    num_models = len(models)
    bar_width = 0.15
    x = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        df_model = df[df['Model'] == model]
        df_model = df_model.set_index("DataSet").reindex(datasets).reset_index()

        y = df_model["MSE"].values
        yerr = df_model["StdDev"].values

        x_pos = x + i * bar_width
        bars = ax.bar(x_pos, y, width=bar_width, label=model, yerr=yerr, capsize=4)

        # Text über jedem Balken
        for k, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # x-Position: Mitte des Balkens
                height + 0.01 * max(y),  # y-Position: etwas über Balken
                f"{height:.2f}",  # Formatierter MSE-Wert
                ha='center', va='bottom', fontsize=8
            )

    ax.set_title(f"Vergleich Modelle")
    ax.set_xlabel("DataSets")
    ax.set_ylabel("MSE")
    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'model_comparison.png')
    plt.savefig(plot_path)
    plt.close()

    # Prozentuale Verbesserung berechnen und speichern
    improvement_results = []
    for dataset in datasets:
        for model in models[1:]:
            # MSE-Werte für beide Modelle abrufen
            mse_nn = df[(df['DataSet'] == dataset) & (df['Model'] == models[0])]['MSE'].values[0]
            mse_rnn = df[(df['DataSet'] == dataset) & (df['Model'] == model)]['MSE'].values[0]

            # Prozentuale Verbesserung berechnen
            improvement = (mse_nn - mse_rnn) / mse_nn * 100
            improvement_results.append((dataset, model, improvement))

    print("\n Modellvergleichsergebnisse:")
    with open(os.path.join(results_dir, 'Results.txt'), 'w') as f:
        f.write("DataSet                 | Model          | MSE        | StdDev\n")
        f.write("-" * 75 + "\n")
        for row in results:
            print(f"{row[0]:<25} | {row[1]:<15} | MSE: {row[2]:.6f} | StdDev: {row[3]:.6f}")
            f.write(f"{row[0]:<25} | {row[1]:<15} | {row[2]:<6} | {row[3]:.6f}\n")

        f.write("\nProzentuale Verbesserung:\n")
        f.write("DataSet                 | Model           | Improvement(%) \n")
        print("DataSet                 | Model           | Improvement(%) \n")
        f.write("-" * 50 + "\n")
        for row in improvement_results:
            print(f"{row[0]:<25} | {row[1]:<15} | {row[2]:.2f}\n")
            f.write(f"{row[0]:<25} | {row[1]:<15} | {row[2]:.2f}\n")

    documentation["Results"] = {
        "Model_Comparison": results,
        "Improvement": improvement_results
    }

    with open(os.path.join(results_dir, 'documentation.json'), 'w') as json_file:
        json.dump(documentation, json_file, indent=4)

    print("\n Ergebnisse wurden in 'documentation.json' und 'Results.txt' gespeichert.")