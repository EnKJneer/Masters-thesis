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
class DataClass_new:
    def __init__(self, name, folder, training_validation_datas, testing_data_paths, target_channels=hdata.HEADER_y, percentage_used=100, load_all_geometrie_variations=True):
        self.name = name
        self.folder = folder
        self.training_validation_datas = training_validation_datas
        self.testing_data_paths = testing_data_paths
        self.target_channels = target_channels
        self.percentage_used = percentage_used
        self.load_all_geometrie_variations = load_all_geometrie_variations

def load_data_new(data_params: DataClass_new, past_values=2, future_values=2, window_size=1, keep_separate=False, N=3):
    """
    Loads and preprocesses data for training, validation, and testing.

    Parameters
    ----------
    data_params : DataClass_new
        A DataClass_new containing the data parameters for loading the dataset.
    past_values : int, optional
        The number of past values to consider. The default is 2.
    future_values : int, optional
        The number of future values to predict. The default is 2.
    window_size : int, optional
        The size of the sliding window. The default is 1.
    keep_separate : bool, optional
        If True, return lists of DataFrames instead of concatenated ones.
    N : int, optional
        Number of packages per file (for train/val split). Default is 3.

    Returns
    -------
    tuple
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if window_size <= 0:
        window_size = 1

    # Load test data
    fulltestdatas = hdata.read_fulldata(data_params.testing_data_paths, data_params.folder)
    test_datas =  hdata.apply_action(fulltestdatas, lambda data: data[hdata.HEADER_x].rolling(window=window_size, min_periods=1).mean())
    X_test =  hdata.apply_action(test_datas, lambda data: hdata.create_full_ml_vector_optimized(past_values, future_values, data))
    test_targets =  hdata.apply_action(fulltestdatas, lambda data: data[data_params.target_channels])
    y_test =  hdata.apply_action(test_targets, lambda target: target.rolling(window=window_size, min_periods=1).mean())
    if past_values + future_values != 0:
        y_test =  hdata.apply_action(y_test, lambda target: target.iloc[past_values:-future_values])

    # Namen der Testdateien zum Ausschluss
    test_files = set(os.path.basename(p) for p in data_params.testing_data_paths)

    # Trainings- und Validierungsdaten vorbereiten
    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []

    for name in data_params.training_validation_datas:
        pattern = f"{name}_*.csv"
        files = glob.glob(os.path.join(data_params.folder, pattern))
        files = [f for f in files if os.path.basename(f) not in test_files]

        file_datas = hdata.read_fulldata(files, data_params.folder)
        file_datas_x =  hdata.apply_action(file_datas, lambda data: data[hdata.HEADER_x].rolling(window=window_size, min_periods=1).mean())
        file_datas_y =  hdata.apply_action(file_datas, lambda data: data[data_params.target_channels].rolling(window=window_size, min_periods=1).mean())

        X_files =  hdata.apply_action(file_datas_x, lambda data: hdata.create_full_ml_vector_optimized(past_values, future_values, data))
        y_files =  hdata.apply_action(file_datas_y, lambda target: target.iloc[past_values:-future_values] if past_values + future_values != 0 else target)

        for X_df, y_df in zip(X_files, y_files):
            X_split = np.array_split(X_df, N)
            y_split = np.array_split(y_df, N)
            # einfache Regel: mittleres Paket = val, Rest = train
            mid = N // 2
            X_val_part = X_split[mid]
            y_val_part = y_split[mid]
            X_train_parts = X_split[:mid] + X_split[mid+1:]
            y_train_parts = y_split[:mid] + y_split[mid+1:]

            all_X_train.extend(X_train_parts)
            all_y_train.extend(y_train_parts)
            all_X_val.append(X_val_part)
            all_y_val.append(y_val_part)

    if keep_separate:
        return all_X_train, all_X_val, X_test, all_y_train, all_y_val, y_test
    else:
        X_train = pd.concat(all_X_train).reset_index(drop=True)
        y_train = pd.concat(all_y_train).reset_index(drop=True)
        X_val = pd.concat(all_X_val).reset_index(drop=True)
        y_val = pd.concat(all_y_val).reset_index(drop=True)
        X_test = pd.concat(X_test).reset_index(drop=True)
        y_test = pd.concat(y_test).reset_index(drop=True)
        return X_train, X_val, X_test, y_train, y_val, y_test

class DataClass:
    def __init__(self, name, folder, training_data_paths, validation_data_paths, testing_data_paths, target_channels = hdata.HEADER_y, percentage_used = 100):
        self.name = name
        self.folder = folder
        self.training_data_paths = training_data_paths
        self.validation_data_paths = validation_data_paths
        self.testing_data_paths = testing_data_paths
        self.target_channels = target_channels
        self.percentage_used = percentage_used
        
def load_data(data_params: DataClass, past_values=2, future_values=2, window_size=1, homogenous_data=False, keep_separate=False):
    """
    Loads and preprocesses data for training, validation, and testing.

    Parameters
    ----------
    data_params : DataClass
        A DataClass containing the data parameters for loading the dataset.
    past_values : int, optional
        The number of past values to consider. The default is 2.
    future_values : int, optional
        The number of future values to predict. The default is 2.
    window_size : int, optional
        The size of the sliding window. The default is 1.
    homogenous_data : bool, optional
        Whether to compute the mean of the training data. The default is False.
        If False, the data will be concatenated.
    keep_separate : bool, optional
        Whether to keep the data separate or combine it. The default is False.

    Returns
    -------
    tuple
        A tuple containing the preprocessed training, validation, and testing data as Pandas DataFrames:
        - X_train : pd.DataFrame
            Preprocessed training data features.
        - y_train : pd.DataFrame
            Preprocessed training data labels.
        - X_val : pd.DataFrame
            Preprocessed validation data features.
        - y_val : pd.DataFrame
            Preprocessed validation data labels.
        - X_test : pd.DataFrame
            Preprocessed test data features.
        - y_test : pd.DataFrame
            Preprocessed test data labels.
    """
    if window_size == 0:
        window_size = 1
    elif window_size < 0:
        window_size = abs(window_size)

    # Getting validation data to right format:
    fulltrainingdatas = hdata.read_fulldata(data_params.training_data_paths, data_params.folder)
    fulltestdatas = hdata.read_fulldata(data_params.testing_data_paths, data_params.folder)
    fullvaldatas = hdata.read_fulldata(data_params.validation_data_paths, data_params.folder)

    # Apply rolling mean to each DataFrame in the lists
    training_datas = hdata.apply_action(fulltrainingdatas, lambda data: data[hdata.HEADER_x].rolling(window=window_size, min_periods=1).mean())
    test_datas = hdata.apply_action(fulltestdatas, lambda data: data[hdata.HEADER_x].rolling(window=window_size, min_periods=1).mean())
    val_datas = hdata.apply_action(fullvaldatas,lambda data: data[hdata.HEADER_x].rolling(window=window_size, min_periods=1).mean())

    # Input data
    X_train = hdata.apply_action(training_datas, lambda data: hdata.create_full_ml_vector_optimized(past_values, future_values, data))
    X_val = hdata.apply_action(val_datas, lambda data: hdata.create_full_ml_vector_optimized(past_values, future_values, data))
    X_test = hdata.apply_action(test_datas, lambda data: hdata.create_full_ml_vector_optimized(past_values, future_values, data))

    # Extract the target columns from each DataFrame in the lists
    training_targets = hdata.apply_action(fulltrainingdatas, lambda data: data[data_params.target_channels])
    test_targets = hdata.apply_action(fulltestdatas, lambda data: data[data_params.target_channels])
    val_targets = hdata.apply_action(fullvaldatas, lambda data: data[data_params.target_channels])

    # Output data
    y_train = hdata.apply_action(training_targets, lambda target: target.rolling(window=window_size, min_periods=1).mean())
    y_val = hdata.apply_action(val_targets, lambda target: target.rolling(window=window_size, min_periods=1).mean())
    y_test = hdata.apply_action(test_targets, lambda target: target.rolling(window=window_size, min_periods=1).mean())

    if past_values + future_values != 0:
        y_train = hdata.apply_action(y_train, lambda target: target.iloc[past_values:-future_values])
        y_val = hdata.apply_action(y_val, lambda target: target.iloc[past_values:-future_values])
        y_test = hdata.apply_action(y_test, lambda target: target.iloc[past_values:-future_values])

    if not keep_separate:
        if not homogenous_data:
            if isinstance(X_train, list):
                X_train = pd.concat(X_train, axis=0).reset_index(drop=True)
                y_train = pd.concat(y_train, axis=0).reset_index(drop=True)
            if isinstance(X_val, list):
                X_val = pd.concat(X_val, axis=0).reset_index(drop=True)
                y_val = pd.concat(y_val, axis=0).reset_index(drop=True)
            if isinstance(X_test, list):
                X_test = pd.concat(X_test, axis=0).reset_index(drop=True)
                y_test = pd.concat(y_test, axis=0).reset_index(drop=True)
        else:
            if isinstance(X_train, list):
                X_train = pd.concat(X_train, axis=0).reset_index(drop=True).mean(axis=0).to_frame().T
                y_train = pd.concat(y_train, axis=0).reset_index(drop=True).mean(axis=0).to_frame().T
            if isinstance(X_val, list):
                X_val = pd.concat(X_val, axis=0).reset_index(drop=True).mean(axis=0).to_frame().T
                y_val = pd.concat(y_val, axis=0).reset_index(drop=True).mean(axis=0).to_frame().T
            if isinstance(X_test, list):
                X_test = pd.concat(X_test, axis=0).reset_index(drop=True).mean(axis=0).to_frame().T
                y_test = pd.concat(y_test, axis=0).reset_index(drop=True).mean(axis=0).to_frame().T

    return X_train, X_val, X_test, y_train, y_val, y_test

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

# Berechne MSE und Standardabweichung pro Modell und Methode
def calculate_mse_and_std(predictions_list, true_values):
    errors = [(pred - true_values) ** 2 for pred in predictions_list]
    mse_values = [np.mean(err) for err in errors]
    return np.mean(mse_values), np.std(mse_values)
""" Data Sets """
folder_data = '..\\..\\DataSets\DataFiltered'
dataSet_same_material_diff_workpiece_new = DataClass_new('Al_Al_Gear_Plate', folder_data,
                                    ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                    ['AL_2007_T4_Plate_Normal_3.csv'],
                                  ["curr_x"],100,)
dataSet_diff_material_same_workpiece_new = DataClass_new('Al_St_Gear_Gear', folder_data,
                                    ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                    ['S235JR_Gear_Normal_3.csv'],
                                  ["curr_x"],100,)
dataSet_diff_material_diff_workpiece_new = DataClass_new('Al_St_Gear_Plate', folder_data,
                                    ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                    ['S235JR_Plate_Normal_3.csv'],
                                  ["curr_x"],100,)
dataSets_list_new = [dataSet_same_material_diff_workpiece_new,dataSet_diff_material_same_workpiece_new,dataSet_diff_material_diff_workpiece_new]

dataSet_same_material_diff_workpiece = DataClass('Al_Al_Gear_Plate', folder_data,
                                    ['AL_2007_T4_Gear_Depth_3.csv','AL_2007_T4_Gear_Normal_1.csv'],
                                    ['AL_2007_T4_Gear_SF_2.csv', 'AL_2007_T4_Gear_Normal_2.csv'],
                                    ['AL_2007_T4_Plate_Normal_3.csv'],
                                  ["curr_x"],100)
dataSet_diff_material_same_workpiece = DataClass('Al_St_Gear_Gear', folder_data,
                                    ['AL_2007_T4_Gear_Depth_3.csv','AL_2007_T4_Gear_Normal_1.csv'],
                                    ['AL_2007_T4_Gear_SF_2.csv', 'AL_2007_T4_Gear_Normal_2.csv'],
                                    ['S235JR_Gear_Normal_3.csv'],
                                  ["curr_x"],100)
dataSet_diff_material_diff_workpiece = DataClass('Al_St_Gear_Plate', folder_data,
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
        X_train_old, X_val_old, X_test_old, y_train_old, y_val_old, y_test_old = load_data(
            data_old, past_values, future_values, window_size)
        X_train_new, X_val_new, X_test_new, y_train_new, y_val_new, y_test_new = load_data_new(
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
