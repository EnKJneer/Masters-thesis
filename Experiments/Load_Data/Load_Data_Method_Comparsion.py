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
            'n_neurons': (15, 128),
            'n_layers': (3, 12),
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

def nn_prediction(data_params, model_params, X_train, y_train, X_val, y_val, X_test, y_test):
    y_predicted = pd.DataFrame()
    input_size = X_train.shape[1]
    axis = data_params.target_channels[0]
    output_size = y_train[axis].T.shape[0] if len(y_train[axis].shape) > 1 else 1

    all_predictions = []
    for i in range(0, NUMBEROFMODELS):
        model_nn = mnn.Net(input_size=input_size, output_size=output_size,
                           n_neurons=model_params['n_neurons'], n_layers=model_params['n_layers'],
                           activation=nn.ReLU)
        val_error = model_nn.train_model(
            X_train, y_train[axis], X_val, y_val[axis],
            learning_rate=model_params['learning_rate'], n_epochs=NUMBEROFEPOCHS, patience=5
        )
        loss, predictions = model_nn.test_model(X_test, y_test[axis])
        all_predictions.append(predictions.flatten())

    mean_predictions = np.mean(all_predictions, axis=0)
    #hplot.plot_prediction_vs_true('Neural_Net ' + data_params.name + ' ' + axis, mean_predictions.T, y_test[axis])
    y_predicted[axis + "_pred"] = mean_predictions
    return y_predicted

def hyperparameter_optimization_rf(folder_path, X_train, X_val, y_train, y_val):
    study_name_rf = "Hyperparameter_RF_mini_"
    default_parameter_rf = {
        'window_size': window_size,
        'past_values': past_values,
        'future_values': future_values,
    }
    num_db_files_rf = sum(file.endswith('.db') for file in os.listdir(folder_path))

    while num_db_files_rf < 4:
        search_space_rf = {
            'n_estimators': (5, 50),
            'max_features': ['sqrt', 'log2', None],
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 4)
        }
        objective_rf = hyperopt.Objective(
            search_space=search_space_rf,
            model=mrf.RandomForestModel,
            data=[X_train, X_val, y_train["curr_x"], y_val["curr_x"]],
            n_epochs=NUMBEROFEPOCHS,
            pruning=True
        )
        hyperopt.optimize(objective_rf, folder_path, study_name_rf, n_trials=NUMBEROFTRIALS)
        hyperopt.WriteAsDefault(folder_path, default_parameter_rf)
        num_db_files_rf = sum(file.endswith('.db') for file in os.listdir(folder_path))

    model_params = hyperopt.GetOptimalParameter(folder_path, plot=False)
    print("Random Forest Hyperparameters:", model_params)
    return model_params

def rf_prediction(data_params, model_params, X_train, y_train, X_val, y_val, X_test, y_test):
    y_predicted = pd.DataFrame()
    for axis in data_params.target_channels:
        all_predictions = []
        model_rf = mrf.RandomForestModel(n_estimators=model_params['n_estimators'],
                                         max_features=model_params['max_features'],
                                         min_samples_leaf=model_params['min_samples_leaf'],
                                         min_samples_split=model_params['min_samples_split'])
        val_error = model_rf.train_model(X_train, y_train[axis], X_val, y_val[axis])
        loss, predictions = model_rf.test_model(X_test, y_test[axis])
        #hplot.plot_prediction_vs_true(name + ' ' + data_params.name + ' ' + axis, predictions.T, y_test[axis])
        y_predicted[axis + "_pred"] = predictions
    return y_predicted

# Berechne MSE und Standardabweichung pro Modell und Methode
def calculate_mse_and_std(predictions_list, true_values):
    errors = [(pred - true_values) ** 2 for pred in predictions_list]
    mse_values = [np.mean(err) for err in errors]
    return np.mean(mse_values), np.std(mse_values)
""" Data Sets """
folder_data = '..\\..\\DataSets\DataFiltered'
dataSet_same_material_diff_workpiece_new = hdata.DataClass_new('Al_Al_Gear_Plate', folder_data,
                                    ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                    ['AL_2007_T4_Plate_Normal_3.csv'],
                                  ["curr_x"],100,)
dataSet_diff_material_same_workpiece_new = hdata.DataClass_new('Al_St_Gear_Gear', folder_data,
                                    ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                    ['S235JR_Gear_Normal_3.csv'],
                                  ["curr_x"],100,)
dataSet_diff_material_diff_workpiece_new = hdata.DataClass_new('Al_St_Gear_Plate', folder_data,
                                    ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                    ['S235JR_Plate_Normal_3.csv'],
                                  ["curr_x"],100,)
dataSets_list_new = [dataSet_same_material_diff_workpiece_new,dataSet_diff_material_same_workpiece_new,dataSet_diff_material_diff_workpiece_new]

dataSet_same_material_diff_workpiece = hdata.DataClass('Al_Al_Gear_Plate', folder_data,
                                    ['AL_2007_T4_Gear_Depth_3.csv','AL_2007_T4_Gear_Normal_1.csv'],
                                    ['AL_2007_T4_Gear_SF_2.csv', 'AL_2007_T4_Gear_Normal_2.csv'],
                                    ['AL_2007_T4_Plate_Normal_3.csv'],
                                  ["curr_x"],100)
dataSet_diff_material_same_workpiece = hdata.DataClass('Al_St_Gear_Gear', folder_data,
                                    ['AL_2007_T4_Gear_Depth_3.csv','AL_2007_T4_Gear_Normal_1.csv'],
                                    ['AL_2007_T4_Gear_SF_2.csv', 'AL_2007_T4_Gear_Normal_2.csv'],
                                    ['S235JR_Gear_Normal_3.csv'],
                                  ["curr_x"],100)
dataSet_diff_material_diff_workpiece = hdata.DataClass('Al_St_Gear_Plate', folder_data,
                                    ['AL_2007_T4_Gear_Depth_3.csv','AL_2007_T4_Gear_Normal_1.csv'],
                                    ['AL_2007_T4_Gear_SF_2.csv', 'AL_2007_T4_Gear_Normal_2.csv'],
                                    ['S235JR_Plate_Normal_3.csv'],
                                  ["curr_x"],100)
dataSets_list = [dataSet_same_material_diff_workpiece,dataSet_diff_material_same_workpiece, dataSet_diff_material_diff_workpiece]

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 10

    window_size = 10
    past_values = 2
    future_values = 2

    #torch.cuda.is_available()

    """Daten für Hyperparameter Optimierung laden"""
    # Load data with old method
    X_train_old, X_val_old, X_test_old, y_train_old, y_val_old, y_test_old = hdata.load_data(dataSets_list[0], past_values=past_values,
                                                                     future_values=future_values,
                                                                     window_size=window_size)

    """ Hyperparameter """
    folder_path_nn = '../../Models/Hyperparameter/NeuralNet_curr_x'
    model_params_nn = hyperparameter_optimization_ml(folder_path_nn, X_train_old, X_val_old, y_train_old, y_val_old)

    folder_path_rf = '../../Models/Hyperparameter/RandomForest_mini'
    model_params_rf = hyperparameter_optimization_rf(folder_path_rf, X_train_old, X_val_old, y_train_old, y_val_old)

    """Save Meta information"""
    # Define the meta information structure
    meta_information = {
        "DataSets": [],
        "Models": {
            "Neural_Net_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                "hyperparameters": {
                    "learning_rate": model_params_nn['learning_rate'],
                    "n_neurons": model_params_nn['n_neurons'],
                    "n_layers": model_params_nn['n_layers']
                }
            },
            "Random_Forest": {
                "hyperparameters": {
                    "n_estimators": model_params_rf['n_estimators'],
                    "max_features": model_params_rf['max_features'],
                    "min_samples_split": model_params_rf['min_samples_split'],
                    "min_samples_leaf": model_params_rf['min_samples_leaf']
                }
            }
        },
        "Data_Preprocessing": {
            "window_size": window_size,
            "past_values": past_values,
            "future_values": future_values
        }
    }
    for data_params in dataSets_list_new:
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
    for i, (data_old, data_new) in enumerate(zip(dataSets_list, dataSets_list_new)):
        data, name = hdata.get_test_data_as_pd(data_old, past_values=past_values, future_values=future_values, window_size=window_size)
        file_path = f'Data/{data_old.name}.csv'
        hdata.save_data(data, file_path)

        print(f"\n===== Verarbeitung: {data_old.name} =====")
        # Daten laden
        X_train_old, X_val_old, X_test_old, y_train_old, y_val_old, y_test_old = hdata.load_data(
            data_old, past_values, future_values, window_size)
        X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new = hdata.load_data_new(
            data_new, past_values, future_values, window_size)

        # Modellvergleich auf alten Daten
        nn_preds_old, rf_preds_old = [], []
        for _ in range(NUMBEROFMODELS):
            model_nn = mnn.Net(X_train_old.shape[1], 1, model_params_nn['n_neurons'],
                               model_params_nn['n_layers'], nn.ReLU)
            model_nn.train_model(X_train_old, y_train_old["curr_x"], X_val_old, y_val_old["curr_x"],
                                 model_params_nn['learning_rate'], NUMBEROFEPOCHS, patience=5)
            _, pred_nn = model_nn.test_model(X_test_old, y_test_old["curr_x"])
            nn_preds_old.append(pred_nn.flatten())

            model_rf = mrf.RandomForestModel(n_estimators=model_params_rf['n_estimators'],
                                             max_features=model_params_rf['max_features'],
                                             min_samples_leaf=model_params_rf['min_samples_leaf'],
                                             min_samples_split=model_params_rf['min_samples_split'])
            model_rf.train_model(X_train_old, y_train_old["curr_x"], X_val_old, y_val_old["curr_x"])
            _, pred_rf = model_rf.test_model(X_test_old, y_test_old["curr_x"])
            rf_preds_old.append(pred_rf.flatten())

        # Modellvergleich auf neuen Daten
        nn_preds_new, rf_preds_new = [], []
        for _ in range(NUMBEROFMODELS):
            model_nn = mnn.Net(X_train_new.shape[1], 1, model_params_nn['n_neurons'],
                               model_params_nn['n_layers'], nn.ReLU)
            model_nn.train_model(X_train_new, y_train_new["curr_x"], X_val_new, y_val_new["curr_x"],
                                 model_params_nn['learning_rate'], NUMBEROFEPOCHS, patience=5)
            _, pred_nn = model_nn.test_model(X_test_new, y_test_new["curr_x"])
            nn_preds_new.append(pred_nn.flatten())

            model_rf = mrf.RandomForestModel(n_estimators=model_params_rf['n_estimators'],
                                             max_features=model_params_rf['max_features'],
                                             min_samples_leaf=model_params_rf['min_samples_leaf'],
                                             min_samples_split=model_params_rf['min_samples_split'])
            model_rf.train_model(X_train_new, y_train_new["curr_x"], X_val_new, y_val_new["curr_x"])
            _, pred_rf = model_rf.test_model(X_test_new, y_test_new["curr_x"])
            rf_preds_new.append(pred_rf.flatten())

        # Fehlerberechnung
        mse_old_nn, std_old_nn = calculate_mse_and_std(nn_preds_old, y_test_old["curr_x"])
        mse_old_rf, std_old_rf = calculate_mse_and_std(rf_preds_old, y_test_old["curr_x"])
        mse_new_nn, std_new_nn = calculate_mse_and_std(nn_preds_new, y_test_new["curr_x"])
        mse_new_rf, std_new_rf = calculate_mse_and_std(rf_preds_new, y_test_new["curr_x"])

        # Ergebnisse speichern
        results.extend([
            [data_old.name, "Neural_Net", "Old", mse_old_nn, std_old_nn],
            [data_old.name, "Neural_Net", "New", mse_new_nn, std_new_nn],
            [data_old.name, "Random_Forest", "Old", mse_old_rf, std_old_rf],
            [data_old.name, "Random_Forest", "New", mse_new_rf, std_new_rf],
        ])
        # Save Results in csv
        header = ["Neural_Net_Old", "Neural_Net_New", "Random_Forest_Old", "Random_Forest_New"]
        data_list = [nn_preds_old, nn_preds_new, rf_preds_old, rf_preds_new]
        data = pd.DataFrame()
        data['y_ground_truth'] = y_test_old["curr_x"]
        for i, col in enumerate(header):
            data[col] = np.mean(data_list[i], axis=0)
        hdata.add_pd_to_csv(file_path, data, header)

        plt.figure(figsize=(12, 6))
        plt.plot(data['y_ground_truth'], label='Ground Truth', color='black', linewidth=2)

        for col in header:
            plt.plot(data[col], linestyle='--', label=f'{col}', alpha=0.6)

        plt.title(f'{data_old.name}: Modellvergleich alt vs. neu')
        plt.xlabel('Zeit')
        plt.ylabel('Strom in A')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()



    # Export als CSV
    # Ordner anlegen
    os.makedirs("Data", exist_ok=True)

    # Ergebnisse ausgeben
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

    # TODO: Anders rum ein Plot pro Model
    # Ein Plot pro Methode
    for i, method in enumerate(methods):
        df_method = df[df['Method'] == method]

        fig, ax = plt.subplots(figsize=(10, 5))

        for j, model in enumerate(models):
            df_model = df_method[df_method['Model'] == model]
            df_model = df_model.set_index("DataSet").reindex(datasets).reset_index()

            y = df_model["MSE"].values
            yerr = df_model["StdDev"].values

            x_pos = x + j * bar_width
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

        ax.set_title(f"Method: {method}")
        ax.set_xlabel("DataSet")
        ax.set_ylabel("MSE")
        ax.set_xticks(x + bar_width * (num_models - 1) / 2)
        ax.set_xticklabels(datasets)
        ax.legend()
        plt.tight_layout()
        plt.show()

    # TODO: Prozentuale Verbesserung speichern
    # Ergebnisse in Textdatei speichern
    with open('Data/model_comparison_results.txt', 'w') as f:
        f.write("DataSet                 | Model           | Method | MSE        | StdDev\n")
        f.write("-" * 75 + "\n")
        for row in results:
            f.write(f"{row[0]:<25} | {row[1]:<15} | {row[2]:<6} | {row[3]:.6f} | {row[4]:.6f}\n")
    print("\n Ergebnisse wurden in 'Data/model_comparison_results.txt' gespeichert.")

