# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:53:02 2024

@author: Jonas Kyrion

SKRIPT DESCRIPTION:
    Contains the functions required for loading and pre-processing the data 
"""
import glob
#libarie import
import os
import pickle
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import anomalies
from scipy import stats
import re

import pandas as pd
import numpy as np
import pickle

# konstanten
WINDOWSIZE = 1

HEADER = ["pos_x", "pos_y", "pos_z", "v_sp", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "a_sp", "curr_x", "curr_y", "curr_z", "curr_sp", "f_x_sim", "f_y_sim", "f_z_sim", "f_sp_sim", "materialremoved_sim"]
HEADER_x = ["v_sp", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "a_sp", "f_x_sim", "f_y_sim", "f_z_sim", "f_sp_sim", "materialremoved_sim"]
HEADER_y = ["curr_x", "curr_y", "curr_z", "curr_sp"]

# Datenklassen
"""
Datensätze bekannt bis Material_Bauteil_unbekannt werden verwendet um zu messen wie die Modelle sich verhalten, wenn gewisses wissen nicht in den Trainingsdaten enthalten ist.
St_Tr_Mat_1 wurde dabei aus dem Datensatz rausgenommen, da es da scheinbar zu einer anomalie gekommen ist.
CMX_St_Val_Mat_2 wird als Testsdatensatz verwendet
"""

class DataClass:
    def __init__(self, name, folder, training_data_paths, validation_data_paths, testing_data_paths, target_channels = HEADER_y, percentage_used = 100):
        self.name = name
        self.folder = folder
        self.training_data_paths = training_data_paths
        self.validation_data_paths = validation_data_paths
        self.testing_data_paths = testing_data_paths
        self.target_channels = target_channels
        self.percentage_used = percentage_used

def create_full_ml_vector_optimized_old(past_values, future_values, channels_in: pd.DataFrame) -> np.array:
    """
    Creates a full machine learning vector optimized for multiple channels.

    Parameters
    ----------
    past_values : int
        The number of past values to consider.
    future_values : int
        The number of future values to predict.
    channels_in : pd.DataFrame
        A DataFrame of input channel data.

    Returns
    -------
    pd.DataFrame
        The optimized machine learning vector for all channels.
    """
    if not isinstance(channels_in, pd.DataFrame):
        channels_in = pd.DataFrame(channels_in).T
    n = len(channels_in)
    full_vector = pd.DataFrame(index=range(n - (past_values + future_values)))

    # Bestimmen der maximalen Länge der Zahlen in den Spaltennamen
    max_digits = len(channels_in.columns)

    for i in range(past_values + future_values + 1):
        if i < past_values:
            shifted = channels_in.shift(-i).iloc[:n - (past_values + future_values), :]
            shifted.columns = [f'{str(col).zfill(max_digits)}_0_past_{i}' for col in channels_in.columns]
        elif i == past_values:
            shifted = channels_in.shift(-past_values).iloc[:n - (past_values + future_values), :]
            shifted.columns = [f'{str(col).zfill(max_digits)}_1_current' for col in channels_in.columns]
        else:
            shifted = channels_in.shift(-i).iloc[:n - (past_values + future_values), :]
            shifted.columns = [f'{str(col).zfill(max_digits)}_2_future_{i - past_values - 1}' for col in channels_in.columns]

        full_vector = pd.concat([full_vector, shifted], axis=1).dropna()

    # Spaltennamen sortieren
    sorted_columns = sorted(full_vector.columns)

    # DataFrame mit sortierten Spaltennamen erstellen
    full_vector = full_vector[sorted_columns]
    return full_vector.to_numpy()

def create_full_ml_vector_optimized(past_values, future_values, channels_in: pd.DataFrame) -> np.array:
    """
    Creates a full machine learning vector optimized for multiple channels.

    Parameters
    ----------
    past_values : int
        The number of past values to consider.
    future_values : int
        The number of future values to predict.
    channels_in : pd.DataFrame
        A DataFrame of input channel data.

    Returns
    -------
    pd.DataFrame
        The optimized machine learning vector for all channels.
    """
    if not isinstance(channels_in, pd.DataFrame):
        channels_in = pd.DataFrame(channels_in).T

    n = len(channels_in)

    # Remove past_values from the beginning and future_values from the end
    #channels_in = channels_in.iloc[past_values:n - future_values]

    full_vector = pd.DataFrame(index=range(n - (past_values + future_values)))

    # Determine the maximum length of the numbers in the column names
    max_digits = len(str(len(channels_in.columns)))

    for i in range(past_values + future_values + 1):
        if i < past_values:
            shifted = channels_in.shift(-i)
            shifted.columns = [f'{str(col).zfill(max_digits)}_0_past_{i}' for col in channels_in.columns]
        elif i == past_values:
            shifted = channels_in.shift(-past_values)
            shifted.columns = [f'{str(col).zfill(max_digits)}_1_current' for col in channels_in.columns]
        else:
            shifted = channels_in.shift(-i)
            shifted.columns = [f'{str(col).zfill(max_digits)}_2_future_{i - past_values - 1}' for col in channels_in.columns]

        full_vector = pd.concat([full_vector, shifted], axis=1).dropna()

    # Sort column names
    sorted_columns = sorted(full_vector.columns)

    # Create DataFrame with sorted column names
    full_vector = full_vector[sorted_columns]
    return full_vector

def read_file(file_path, header = HEADER):
    """
    Reads .csv .pkl, and .npz files and returns them as a Pandas DataFrame.

    :param file_path: Path to the file
    :return: Pandas DataFrame containing the data from the file
    """
    if file_path.endswith('.csv'):
        # Directly read CSV files into a DataFrame
        return pd.read_csv(file_path)

    elif file_path.endswith('.npz'):
        # ToDo: Header anpassen, npz waren vermutlich angeordnet wie pkl
        # Load .npz files and concatenate arrays if there are multiple
        with np.load(file_path) as data:
            if len(data.files) > 1:
                arrays = [data[key] for key in data.files]
                combined_array = np.concatenate(arrays, axis=0)
            else:
                combined_array = data[data.files[0]]

            # Convert the NumPy array to a Pandas DataFrame
            df = pd.DataFrame(combined_array)
            df.columns = header
            return df

    elif file_path.endswith('.pkl'):
        # ToDo: Header anpassen, pkl waren anders angeordnet
        # ToDo: pkl haben keine v und a muss berechnet werden
        # Load .pkl files and convert the data to a DataFrame
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            df = pd.DataFrame(data)
            df.columns = header
            return df
    else:
        raise ValueError("Invalid file format. Please provide a CSV, NPZ, or PKL file.")

def read_fulldata(names, folder):
    """
    Reads full data from multiple files and returns it as a list of DataFrames.

    Parameters
    ----------
    names : list of str
        A list of file names to read.

    folder : str, optional
        The folder where the data is located.

    Returns
    -------
    pd.DataFrame or list of pd.DataFrame
        The full data read from the files, either as a single DataFrame or a list of DataFrames.
    """
    results = []

    for name in names:
        file_path = os.path.join(folder, name)
        if os.path.exists(file_path):
            fulldata = read_file(file_path)
        else:
            print(f"ERROR: No file for {file_path} was found")
            fulldata = None

        if fulldata is not None:
            results.append(fulldata)

    return results

def apply_action(data, action):
    """
    Apply a given action to each item in a list or a single item.

    Parameters:
    data (list or any): The input data. If it's a list, the action will be applied to each item.
                        If it's not a list, the action will be applied to the single item directly.
    action (function): The action to be applied to each item in the data.

    Returns:
    list or any: The result of applying the action to each item in the data. If the input data was a list,
                 the output will be a list. If the input data was not a list, the output will be the result
                 of applying the action to the single item directly.
    """
    if isinstance(data, list) or data.ndim == 3:
        return [action(item) for item in data]
    else:
        return action(data)

def replace_outliners(data, threshold=10):
    # Calculate z-scores
    z_scores = np.abs(stats.zscore(data))

    # Create a copy of the data to work with
    data_copy = data.copy()

    # Calculate z-scores
    z_scores = np.abs(stats.zscore(data))

    # Replace outliers with the standard deviation multiplied by the threshold
    data_copy[z_scores > threshold] = 0
    return data_copy

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
    fulltrainingdatas = read_fulldata(data_params.training_data_paths, data_params.folder)
    fulltestdatas = read_fulldata(data_params.testing_data_paths, data_params.folder)
    fullvaldatas = read_fulldata(data_params.validation_data_paths, data_params.folder)

    # Apply rolling mean to each DataFrame in the lists
    training_datas = apply_action(fulltrainingdatas, lambda data: data[HEADER_x].rolling(window=window_size, min_periods=1).mean())
    test_datas = apply_action(fulltestdatas, lambda data: data[HEADER_x].rolling(window=window_size, min_periods=1).mean())
    val_datas = apply_action(fullvaldatas,lambda data: data[HEADER_x].rolling(window=window_size, min_periods=1).mean())

    # Input data
    X_train = apply_action(training_datas, lambda data: create_full_ml_vector_optimized(past_values, future_values, data))
    X_val = apply_action(val_datas, lambda data: create_full_ml_vector_optimized(past_values, future_values, data))
    X_test = apply_action(test_datas, lambda data: create_full_ml_vector_optimized(past_values, future_values, data))

    # Extract the target columns from each DataFrame in the lists
    training_targets = apply_action(fulltrainingdatas, lambda data: data[data_params.target_channels])
    test_targets = apply_action(fulltestdatas, lambda data: data[data_params.target_channels])
    val_targets = apply_action(fullvaldatas, lambda data: data[data_params.target_channels])

    # Output data
    y_train = apply_action(training_targets, lambda target: target.rolling(window=window_size, min_periods=1).mean())
    y_val = apply_action(val_targets, lambda target: target.rolling(window=window_size, min_periods=1).mean())
    y_test = apply_action(test_targets, lambda target: target.rolling(window=window_size, min_periods=1).mean())

    if past_values + future_values != 0:
        y_train = apply_action(y_train, lambda target: target.iloc[past_values:-future_values])
        y_val = apply_action(y_val, lambda target: target.iloc[past_values:-future_values])
        y_test = apply_action(y_test, lambda target: target.iloc[past_values:-future_values])

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

def load_filtered_data(data_params: DataClass, past_values=2, future_values=2, window_size=1):
    """
    Loads and preprocesses data for training, validation, and testing.
    Does not create ml vector.
    No separation between x and y.

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

    Returns
    -------
    tuple
        A tuple containing the preprocessed training, validation, and testing data as Pandas DataFrames:
        - training_datas : pd.DataFrame
            Preprocessed training data.
        - val_datas : pd.DataFrame
            Preprocessed validation data.
        - test_datas : pd.DataFrame
            Preprocessed test data.
    """
    if window_size == 0:
        window_size = 1
    elif window_size < 0:
        window_size = abs(window_size)

    # Getting validation data to right format:
    fulltrainingdatas = read_fulldata(data_params.training_data_paths, data_params.folder)
    fulltestdatas = read_fulldata(data_params.testing_data_paths, data_params.folder)
    fullvaldatas = read_fulldata(data_params.validation_data_paths, data_params.folder)

    # Apply rolling mean to each DataFrame in the lists
    training_datas = apply_action(fulltrainingdatas, lambda data: data[HEADER].rolling(window=window_size, min_periods=1).mean())
    test_datas = apply_action(fulltestdatas, lambda data: data[HEADER].rolling(window=window_size, min_periods=1).mean())
    val_datas = apply_action(fullvaldatas,lambda data: data[HEADER].rolling(window=window_size, min_periods=1).mean())

    if past_values + future_values != 0:
        training_datas = apply_action(training_datas, lambda target: target.iloc[past_values:-future_values])
        val_datas = apply_action(val_datas, lambda target: target.iloc[past_values:-future_values])
        test_datas = apply_action(test_datas, lambda target: target.iloc[past_values:-future_values])

    base_names = []
    for path in data_params.testing_data_paths:
        file_name = os.path.basename(path)
        name_without_extension = os.path.splitext(file_name)[0]
        # Split the name by underscore and remove the last part (the index)
        base_names.append('_'.join(name_without_extension.split('_')[:-1]))
    return training_datas, val_datas, test_datas, base_names

def extract_base_names(file_names):
    """
    Extracts the base names from a list of file names, removing the index and file extension.

    Parameters
    ----------
    file_names : list of str
        A list of file names with the format 'name_example_index.extension'.

    Returns
    -------
    list of str
        A list of base names without the index and file extension.
    """
    base_names = set()

    for file_name in file_names:
        # Remove the file extension
        name_without_extension = os.path.splitext(file_name)[0]

        # Split the name by underscore and remove the last part (the index)
        base_name = '_'.join(name_without_extension.split('_')[:-1])

        # Add the base name to the set
        base_names.add(base_name)

    return list(base_names)
def get_csv_files(folder_path):
    """
    Retrieves the names of all .csv files in the specified folder.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing the .csv files.

    Returns
    -------
    list of str
        A list of .csv file names in the specified folder.
    """
    # Use glob to find all .csv files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # Extract the file names from the full paths
    csv_file_names = [os.path.basename(file) for file in csv_files]

    return csv_file_names
def count_base_names(file_names):
    """
    Counts the occurrences of base names in a list of file names.

    Parameters
    ----------
    file_names : list of str
        A list of file names with the format 'name_example_index.extension'.

    Returns
    -------
    dict
        A dictionary with base names as keys and their occurrence counts as values.
    """
    base_names = []

    for file_name in file_names:
        # Remove the file extension
        name_without_extension = os.path.splitext(file_name)[0]

        # Split the name by underscore and remove the last part (the index)
        base_name = '_'.join(name_without_extension.split('_')[:-1])

        # Add the base name to the list
        base_names.append(base_name)

    # Count occurrences of each base name
    base_name_counts = Counter(base_names)

    return dict(base_name_counts)
def check_anomali_in_filename(anomalie_replacements, file_name):
    """
    Checks if any "anomali" value from the list of dictionaries is present in the file name.

    Parameters
    ----------
    anomalie_replacements : list of dict
        A list of dictionaries with "anomali" and "normal" keys.
    file_name : str
        The file name to check.

    Returns
    -------
    bool
        True if any "anomali" value is found in the file name, False otherwise.
    """
    for entry in anomalie_replacements:
        anomali_value = entry.get("anomali")
        if anomali_value and anomali_value in file_name:
            return True
    return False
def replace_anomali_with_normal(anomalie_replacements, file_name):
    """
    Replaces substrings in the file name based on "anomali" and "normal" values from a list of dictionaries.

    Parameters
    ----------
    anomalie_replacements : list of dict
        A list of dictionaries with "anomali" and "normal" keys.
    file_name : str
        The file name to perform replacements on.

    Returns
    -------
    str
        The file name with replacements made.
    """
    for entry in anomalie_replacements:
        anomali_value = entry.get("anomali")
        normal_value = entry.get("normal")
        if anomali_value and anomali_value in file_name:
            file_name = file_name.replace(anomali_value, normal_value)
    return file_name

def get_DataClasses_old(folder_path, anomalie_replacements = [{"anomali": 'Blowhole', "normal": 'Normal'}]):
    file_names = get_csv_files(folder_path)
    base_names = extract_base_names(file_names)
    base_name_counts = count_base_names(file_names)

    data_classes = []

    for name in base_names:
        # Check if enough versions exist
        if base_name_counts[name] >= 3:
            # check if data contains anomalie
            if anomalie_replacements is None or check_anomali_in_filename(anomalie_replacements, name):
                training_name = [name + '_1.csv']
                validation_name = [name + '_2.csv']
                test_name = [name + '_3.csv']

            else:
                # training and validation data should be replaced with normal data
                new_name = replace_anomali_with_normal(anomalie_replacements, name)
                training_name = [new_name + '_1.csv']
                validation_name = [new_name + '_2.csv']
                test_name = [name + '_3.csv']

            data_classes.append(DataClass(name, folder_path, training_name, validation_name, test_name))

    return data_classes

def get_material_lists(file_names, materials):
    filtered = {material: [name for name in file_names if material in name] for material in materials}
    filtered['others'] = [name for name in file_names if not any(material in name for material in materials)]
    return filtered

def filter_out_file_names(file_names, filters):
    """
    This function returns a list of file names that do not contain any of the specified substrings.

    :param file_names: List of file names to filter.
    :param filters: List of substrings to filter out.
    :return: List of file names that do not contain any of the specified substrings.
    """
    return [file_name for file_name in file_names if not any(filter in file_name for filter in filters)]


def get_DataClasses_material_spereated(folder_path, materials = ['AL_2007_T4', 'S235JR'], anomalies = ['Blowhole', 'Ano', 'Fehler']):
    file_names = get_csv_files(folder_path)

    files_material_depended = get_material_lists(file_names, materials)
    materials.append('others')
    data_classes = []
    for material in materials:
        files = files_material_depended[material]
        base_names = extract_base_names(files)
        for base_name in base_names:
            if any(anomalie in base_name for anomalie in anomalies):
                filter = anomalies
                filter.append(base_name)
            else:
                filter = [base_name]
            filtered_file_names = filter_out_file_names(files, filter)
            training_name = filtered_file_names[0::2]
            validation_name = filtered_file_names[1::2]
            test_name = [base_name + '_2.csv']
            data_classes.append(DataClass(base_name, folder_path, training_name, validation_name, test_name))

    return data_classes

def get_DataClasses(folder_path, anomalies = ['Blowhole', 'Ano', 'Fehler']):
    file_names = get_csv_files(folder_path)
    base_names = extract_base_names(file_names)
    data_classes = []
    for base_name in base_names:
        if any(anomalie in base_name for anomalie in anomalies):
            filter = anomalies
            filter.append(base_name)
        else:
            filter = [base_name]
        filtered_file_names = filter_out_file_names(file_names, filter)
        training_name = filtered_file_names[0::2]
        validation_name = filtered_file_names[1::2]
        test_name = [base_name + '_2.csv']
        data_classes.append(DataClass(base_name, folder_path, training_name, validation_name, test_name))

    return data_classes

def create_DataClasses_from_base_names(folder_path, training_base_names, validation_base_names, test_base_names, name):
    file_names = get_csv_files(folder_path)

    training_name = [file_name for file_name in file_names if any(name in file_name for name in training_base_names)]
    validation_name = [file_name for file_name in file_names if any(name in file_name for name in validation_base_names)]
    test_name = [file_name for file_name in file_names if any(name in file_name for name in test_base_names)]

    return DataClass(name, folder_path, training_name, validation_name, test_name)