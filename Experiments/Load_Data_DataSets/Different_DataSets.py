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

    toggle = True  # Start mit gerade = Train

    for name in data_params.training_validation_datas:
        pattern = f"{name}_*.csv"
        files = glob.glob(os.path.join(data_params.folder, pattern))
        files = [f for f in files if os.path.basename(f) not in test_files]

        file_datas = hdata.read_fulldata(files, data_params.folder)
        file_datas_x = hdata.apply_action(file_datas, lambda data: data[hdata.HEADER_x].rolling(window=window_size,
                                                                                                min_periods=1).mean())
        file_datas_y = hdata.apply_action(file_datas,
                                          lambda data: data[data_params.target_channels].rolling(window=window_size,
                                                                                                 min_periods=1).mean())

        X_files = hdata.apply_action(file_datas_x,
                                     lambda data: hdata.create_full_ml_vector_optimized(past_values, future_values,
                                                                                        data))
        y_files = hdata.apply_action(file_datas_y, lambda target: target.iloc[
                                                                  past_values:-future_values] if past_values + future_values != 0 else target)

        for X_df, y_df in zip(X_files, y_files):
            X_split = np.array_split(X_df, N)
            y_split = np.array_split(y_df, N)

            if toggle:
                train_indices = [i for i in range(N) if i % 2 == 0]
                val_indices = [i for i in range(N) if i % 2 != 0]
            else:
                train_indices = [i for i in range(N) if i % 2 != 0]
                val_indices = [i for i in range(N) if i % 2 == 0]

            X_train_parts = [X_split[i].reset_index(drop=True) for i in train_indices]
            y_train_parts = [y_split[i].reset_index(drop=True) for i in train_indices]
            X_val_parts = [X_split[i].reset_index(drop=True) for i in val_indices]
            y_val_parts = [y_split[i].reset_index(drop=True) for i in val_indices]

            X_train_parts, y_train_parts = zip(*[hdata.preprocessing(X, y, 12) for X, y in zip(X_train_parts, y_train_parts)])
            X_val_parts, y_val_parts = zip(*[hdata.preprocessing(X, y, 12) for X, y in zip(X_val_parts, y_val_parts)])

            all_X_train.extend(X_train_parts)
            all_y_train.extend(y_train_parts)
            all_X_val.extend(X_val_parts)
            all_y_val.extend(y_val_parts)

        toggle = not toggle  # Umschalten für nächste Datei

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

    X_train, y_train = hdata.preprocessing(X_train, y_train, 12)
    X_val, y_val = hdata.preprocessing(X_val, y_val, 12)
    return X_train, X_val, X_test, y_train, y_val, y_test



""" Data Sets """
folder_data = '..\\..\\DataSets\DataFiltered'

dataPaths_Test = [  'AL_2007_T4_Gear_Normal_3.csv','AL_2007_T4_Plate_Normal_3.csv', 'S235JR_Gear_Normal_3.csv','S235JR_Plate_Normal_3.csv']

dataPaths_Train = [ 'AL_2007_T4_Gear_SF_1.csv', 'AL_2007_T4_Gear_Depth_1.csv','AL_2007_T4_Gear_Normal_1.csv',
                    'AL_2007_T4_Gear_SF_2.csv', 'AL_2007_T4_Gear_Depth_2.csv','AL_2007_T4_Gear_Normal_2.csv',]

dataPaths_Val = [   'Kühlgrill_Mat_S2800_1.csv', 'Kühlgrill_Mat_S3800_1.csv','Kühlgrill_Mat_S4700_1.csv', 'Laufrad_Durchlauf_1_1.csv', 'Laufrad_Durchlauf_2_1.csv',
                    'Kühlgrill_Mat_S2800_2.csv', 'Kühlgrill_Mat_S3800_2.csv','Kühlgrill_Mat_S4700_1.csv', 'Laufrad_Durchlauf_1_2.csv',  'Laufrad_Durchlauf_2_2.csv',
                    'Kühlgrill_Mat_S2800_3.csv', 'Kühlgrill_Mat_S3800_3.csv','Kühlgrill_Mat_S4700_1.csv', 'Laufrad_Durchlauf_1_3.csv', 'Laufrad_Durchlauf_2_3.csv']

Combined_Gear = hdata.DataClass('Gear', folder_data,
                                    dataPaths_Train,
                                    dataPaths_Val,
                                    dataPaths_Test,
                                  ["curr_x"],100)
Combined_KL = hdata.DataClass('KL', folder_data,
                                    dataPaths_Val,
                                    dataPaths_Train,
                                    dataPaths_Test,
                                  ["curr_x"],100)

Combined_Gear_new = hdata.DataClass_CombinedTrainVal('Gear_TrainVal', folder_data,
                                    ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                    dataPaths_Test,
                                  ["curr_x"],100,)
dataSets_list = [Combined_Gear]
dataSets_list_new = [Combined_Gear_new]
if __name__ == "__main__":
    """ Constants """
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 2

    window_size = 10
    past_values = 2
    future_values = 2

    model_rf = mrf.get_reference()

    dataSets_list = [Combined_Gear, Combined_Gear_new, Combined_KL]

    hexp.run_experiment(dataSets_list, True, False, [model_rf], NUMBEROFEPOCHS, NUMBEROFMODELS, past_values, future_values)

