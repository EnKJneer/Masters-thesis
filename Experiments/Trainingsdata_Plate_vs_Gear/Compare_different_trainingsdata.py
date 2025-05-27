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
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
import matplotlib.pyplot as plt
import seaborn as sns

""" Functions """

""" Data Sets """
fgearer_data = '..\\..\\DataSets\DataFiltered'
Al_Al_Gear_Plate = hdata.DataClass_CombinedTrainVal('Al_Al_Gear_Plate', fgearer_data,
                                                    ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                                    ['AL_2007_T4_Plate_Normal_3.csv'],
                                                    ["curr_x"], 100, )
Al_St_Gear_Gear = hdata.DataClass_CombinedTrainVal('Al_St_Gear_Gear', fgearer_data,
                                                   ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                                   ['S235JR_Gear_Normal_3.csv'],
                                                   ["curr_x"], 100, )
Al_St_Gear_Plate = hdata.DataClass_CombinedTrainVal('Al_St_Gear_Plate', fgearer_data,
                                                    ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                                    ['S235JR_Plate_Normal_3.csv'],
                                                    ["curr_x"], 100, )
dataSets_list_Gear = [Al_Al_Gear_Plate,Al_St_Gear_Gear,Al_St_Gear_Plate]

Al_Al_Plate_Gear = hdata.DataClass_CombinedTrainVal('Al_Al_Plate_Gear', fgearer_data,
                                                    ['AL_2007_T4_Plate', 'AL_2007_T4_Plate_Depth', 'AL_2007_T4_Plate_SF'],
                                                    ['AL_2007_T4_Gear_Normal_3.csv'],
                                                    ["curr_x"], 100, )
Al_St_Plate_Plate = hdata.DataClass_CombinedTrainVal('Al_St_Plate_Plate', fgearer_data,
                                                     ['AL_2007_T4_Plate', 'AL_2007_T4_Plate_Depth', 'AL_2007_T4_Plate_SF'],
                                                     ['S235JR_Plate_Normal_3.csv'],
                                                     ["curr_x"], 100, )
Al_St_Plate_Gear = hdata.DataClass_CombinedTrainVal('Al_St_Plate_Gear', fgearer_data,
                                                    ['AL_2007_T4_Plate', 'AL_2007_T4_Plate_Depth', 'AL_2007_T4_Plate_SF'],
                                                    ['S235JR_Gear_Normal_3.csv'],
                                                    ["curr_x"], 100, )
dataSets_list_Plate = [Al_Al_Plate_Gear,Al_St_Plate_Plate,Al_St_Plate_Gear]

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 10

    window_size = 10
    past_values = 2
    future_values = 2

    # torch.cuda.is_available()
    # Load data with gear method --> Needed to get input size
    X_train_gear, X_val_gear, X_test_gear, y_train_gear, y_val_gear, y_test_gear = hdata.load_data(dataSets_list_Gear[0],
                                                                                       past_values=past_values,
                                                                                       future_values=future_values,
                                                                                       window_size=window_size)

    model_nn = mnn.get_reference(X_train_gear.shape[1])
    model_rf = mrf.get_reference_model()
    """Save Meta information"""
    # Define the meta information structure
    meta_information = {
        "DataSets": [],
        "Models": {
            "Neural_Net_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                "hyperparameters": {
                    "learning_rate": model_nn.learning_rate,
                    "n_hidden_size": model_nn.n_hidden_size,
                    "n_hidden_layers": model_nn.n_hidden_layers,
                }
            },
            "Random_Forest": {
                "hyperparameters": {
                    "n_estimators": model_rf.n_estimators,
                    "max_features": model_rf.max_features,
                    "min_samples_split": model_rf.min_samples_split,
                    "min_samples_leaf": model_rf.min_samples_leaf
                }
            }
        },
        "Data_Preprocessing": {
            "window_size": window_size,
            "past_values": past_values,
            "future_values": future_values
        }
    }
    for dataSets_list in [dataSets_list_Plate, dataSets_list_Gear]:
        for data_params in dataSets_list:
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
    with open('Data/info.json', 'w') as json_file:
        json.dump(meta_information, json_file, indent=4)

    """ Prediction """
    results = []
    for i, (data_gear, data_plate) in enumerate(zip(dataSets_list_Gear, dataSets_list_Plate)):
        df_gear, name = hdata.get_test_data_as_pd(data_gear, past_values=past_values, future_values=future_values,
                                               window_size=window_size)
        file_path = f'Data/{data_gear.name}.csv'
        hdata.save_data([df_gear], [file_path])
        df_plate, name = hdata.get_test_data_as_pd(data_plate, past_values=past_values, future_values=future_values,
                                               window_size=window_size)
        file_path = f'Data/{data_gear.name}.csv'
        hdata.save_data([df_plate], [file_path])

        print(f"\n===== Verarbeitung: {data_gear.name} =====")
        # Daten laden
        X_train_gear, X_val_gear, X_test_gear, y_train_gear, y_val_gear, y_test_gear = hdata.load_data(
            data_gear, past_values, future_values, window_size)
        X_train_plate, X_val_plate, X_test_plate, y_train_plate, y_val_plate, y_test_plate = hdata.load_data(
            data_plate, past_values, future_values, window_size)

        # Modellvergleich auf alten Daten
        nn_preds_gear, rf_preds_gear = [], []
        for _ in range(NUMBEROFMODELS):
            model_nn.train_model(X_train_gear, y_train_gear["curr_x"], X_val_gear, y_val_gear["curr_x"], NUMBEROFEPOCHS,
                                 patience=5)
            _, pred_nn = model_nn.test_model(X_test_gear, y_test_gear["curr_x"])
            nn_preds_gear.append(pred_nn.flatten())

            model_rf.train_model(X_train_gear, y_train_gear["curr_x"], X_val_gear, y_val_gear["curr_x"])
            _, pred_rf = model_rf.test_model(X_test_gear, y_test_gear["curr_x"])
            rf_preds_gear.append(pred_rf.flatten())

        # Modellvergleich auf neuen Daten
        nn_preds_plate, rf_preds_plate = [], []
        for _ in range(NUMBEROFMODELS):
            model_nn.train_model(X_train_plate, y_train_plate["curr_x"], X_val_plate, y_val_plate["curr_x"], NUMBEROFEPOCHS,
                                 patience=5)
            _, pred_nn = model_nn.test_model(X_test_plate, y_test_plate["curr_x"])
            nn_preds_plate.append(pred_nn.flatten())

            model_rf.train_model(X_train_plate, y_train_plate["curr_x"], X_val_plate, y_val_plate["curr_x"])
            _, pred_rf = model_rf.test_model(X_test_plate, y_test_plate["curr_x"])
            rf_preds_plate.append(pred_rf.flatten())

        # Fehlerberechnung
        n_drop_values = 10
        mse_gear_nn, std_gear_nn = hdata.calculate_mse_and_std(nn_preds_gear, y_test_gear["curr_x"], n_drop_values, center_data=True)
        mse_gear_rf, std_gear_rf = hdata.calculate_mse_and_std(rf_preds_gear, y_test_gear["curr_x"], n_drop_values, center_data=True)
        mse_plate_nn, std_plate_nn = hdata.calculate_mse_and_std(nn_preds_plate, y_test_plate["curr_x"], n_drop_values, center_data=True)
        mse_plate_rf, std_plate_rf = hdata.calculate_mse_and_std(rf_preds_plate, y_test_plate["curr_x"], n_drop_values, center_data=True)

        # Ergebnisse speichern
        results.extend([
            [data_gear.name, "Neural_Net", "Gear", mse_gear_nn, std_gear_nn],
            [data_plate.name, "Neural_Net", "Plate", mse_plate_nn, std_plate_nn],
            [data_gear.name, "Random_Forest", "Gear", mse_gear_rf, std_gear_rf],
            [data_plate.name, "Random_Forest", "Plate", mse_plate_rf, std_plate_rf],
        ])
        # Save Results in csv
        header = ["Neural_Net_gear", "Random_Forest_gear", ]
        data_list = [nn_preds_gear, rf_preds_gear]
        data = pd.DataFrame()
        data['y_ground_truth'] = y_test_gear["curr_x"]
        for i, col in enumerate(header):
            data[col] = np.mean(data_list[i], axis=0)
        hdata.add_pd_to_csv(file_path, data, header)

        plt.figure(figsize=(12, 6))
        plt.plot(data['y_ground_truth'], label='Ground Truth', color='black', linewidth=2)

        for col in header:
            plt.plot(data[col], linestyle='--', label=f'{col}', alpha=0.6)

        plt.title(f'{data_gear.name}: Modellvergleich gear vs. plate')
        plt.xlabel('Zeit')
        plt.ylabel('Strom in A')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Save Results in csv
        header = ["Neural_Net_plate", "Random_Forest_plate"]
        data_list = [nn_preds_plate, rf_preds_plate]
        data = pd.DataFrame()
        data['y_ground_truth'] = y_test_plate["curr_x"]
        for i, col in enumerate(header):
            data[col] = np.mean(data_list[i], axis=0)
        hdata.add_pd_to_csv(file_path, data, header)

        plt.figure(figsize=(12, 6))
        plt.plot(data['y_ground_truth'], label='Ground Truth', color='black', linewidth=2)

        for col in header:
            plt.plot(data[col], linestyle='--', label=f'{col}', alpha=0.6)

        plt.title(f'{data_plate.name}: Modellvergleich gear vs. plate')
        plt.xlabel('Zeit')
        plt.ylabel('Strom in A')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Export als CSV
    # Ordner anlegen
    os.makedirs("Data", exist_ok=True)

    print("\n Modellvergleichsergebnisse:")
    for row in results:
        print(f"{row[0]:<25} | {row[1]:<15} | {row[2]:<5} | MSE: {row[3]:.6f} | StdDev: {row[4]:.6f}")

    df = pd.DataFrame(results, columns=["DataSet", "Model", "Method", "MSE", "StdDev"])
    methods = df['Method'].unique()
    models = df['Model'].unique()
    datasets = sorted(df['DataSet'].unique())

    num_methods = len(methods)
    num_models = len(models)
    bar_width = 0.15
    x = np.arange(len(datasets))

    # Ein Plot pro Modell
    for i, model in enumerate(models):
        df_model = df[df['Model'] == model]

        fig, ax = plt.subplots(figsize=(10, 5))

        for j, method in enumerate(methods):
            df_method = df_model[df_model['Method'] == method]
            df_method = df_method.set_index("DataSet").reindex(datasets).reset_index()

            y = df_method["MSE"].values
            yerr = df_method["StdDev"].values

            x_pos = x + j * bar_width
            bars = ax.bar(x_pos, y, width=bar_width, label=method, yerr=yerr, capsize=4)

            # Text über jedem Balken
            for k, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # x-Position: Mitte des Balkens
                    height + 0.01 * max(y),  # y-Position: etwas über Balken
                    f"{height:.2f}",  # Formatierter MSE-Wert
                    ha='center', va='bottom', fontsize=8
                )

        ax.set_title(f"Model: {model}")
        ax.set_xlabel("DataSet")
        ax.set_ylabel("MSE")
        ax.set_xticks(x + bar_width * (num_methods - 1) / 2)
        ax.set_xticklabels(datasets)
        ax.legend()
        plt.tight_layout()
        plt.show()

    # Prozentuale Verbesserung berechnen und speichern
    improvement_results = []
    for dataset in datasets:
        for model in models:
            df_dataset_model = df[(df['DataSet'] == dataset) & (df['Model'] == model)]
            if len(df_dataset_model) > 1:
                mse_values = df_dataset_model['MSE'].values
                improvement = (mse_values[0] - mse_values[1]) / mse_values[0] * 100
                improvement_results.append((dataset, model, improvement))

    with open('Data/model_comparison_results.txt', 'w') as f:
        f.write("DataSet                 | Model           | Method | MSE        | StdDev\n")
        f.write("-" * 75 + "\n")
        for row in results:
            f.write(f"{row[0]:<25} | {row[1]:<15} | {row[2]:<6} | {row[3]:.6f} | {row[4]:.6f}\n")

        f.write("\nProzentuale Verbesserung:\n")
        f.write("DataSet                 | Model           | Improvement(%) \n")
        f.write("-" * 50 + "\n")
        for row in improvement_results:
            f.write(f"{row[0]:<25} | {row[1]:<15} | {row[2]:.2f}\n")

    print("\n Ergebnisse wurden in 'Data/model_comparison_results.txt' gespeichert.")
