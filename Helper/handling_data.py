# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:53:02 2024

@author: Jonas Kyrion

SKRIPT DESCRIPTION:
    Contains the functions required for loading and pre-processing the data 
"""
import os
from collections import Counter, deque
import pandas as pd
import numpy as np
import pickle
from abc import ABC, abstractmethod
from scipy.signal import butter, filtfilt

# konstanten
WINDOWSIZE = 1

HEADER = ["pos_x", "pos_y", "pos_z", "v_sp", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "a_sp", "curr_x", "curr_y", "curr_z", "curr_sp", "f_x_sim", "f_y_sim", "f_z_sim", "f_sp_sim", "materialremoved_sim"]
HEADER_x = ["v_sp", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "a_sp", "f_x_sim", "f_y_sim", "f_z_sim", "f_sp_sim", "materialremoved_sim"]
HEADER_v = ["v_sp", "v_x", "v_y", "v_z"]
HEADER_y = ["curr_x", "curr_y", "curr_z", "curr_sp"]

class BaseDataClass(ABC):
    @abstractmethod
    def __init__(self, name, folder, testing_data_paths, target_channels=HEADER_y, past_values=0, future_values=0, window_size=1, add_padding=False):
        self.name = name
        self.folder = folder
        self.target_channels = target_channels

        self.past_values = past_values
        self.future_values = future_values
        if window_size == 0:
            self.window_size = 1
        elif window_size < 0:
            self.window_size = abs(self.window_size)
        else:
            self.window_size = window_size

        self.testing_data_paths = testing_data_paths

        self.add_padding = add_padding

    @staticmethod
    def create_padding(df, length=10):
        """
        Erstellt ein Padding-DataFrame mit Nullen.

        Parameters
        ----------
        df : pandas.DataFrame
            Ein DataFrame aus der Liste, um die Spaltennamen zu erhalten.
        length : int, optional
            Die Länge des Padding-DataFrames (Standard ist 10).

        Returns
        -------
        pandas.DataFrame
            Ein DataFrame mit Nullen.
        """
        return pd.DataFrame(0, index=range(length), columns=df.columns)
    @staticmethod
    def insert_padding(dataframes):
        """
        Fügt Padding zwischen den DataFrames in der Liste ein.

        Parameters
        ----------
        dataframes : list of pandas.DataFrame
            Eine Liste von DataFrames, zwischen denen das Padding eingefügt werden soll.

        Returns
        -------
        pandas.DataFrame
            Ein einzelner DataFrame, der alle DataFrames mit Padding enthält.
        """
        padded_list = []
        for df in dataframes:
            padded_list.append(df)
            padded_list.append(BaseDataClass.create_padding(df))
        padded_list.pop()  # Entferne das letzte Padding
        return pd.concat(padded_list).reset_index(drop=True)

    def prepare_output(self, all_X_train, all_X_val, X_test, all_y_train, all_y_val, y_test):
        """
        Kombiniert und bereitet die Trainings-, Validierungs- und Testdaten vor.

        Parameters
        ----------
        all_X_train : list or pandas.DataFrame
            Trainingsdaten.
        all_X_val : list or pandas.DataFrame
            Validierungsdaten.
        X_test : pandas.DataFrame
            Testdaten.
        all_y_train : list or pandas.DataFrame
            Trainingslabels.
        all_y_val : list or pandas.DataFrame
            Validierungslabels.
        y_test : pandas.DataFrame
            Testlabels.

        Returns
        -------
        tuple
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Sicherstellen, dass die Eingaben Listen sind
        all_X_train = [all_X_train] if not isinstance(all_X_train, list) else all_X_train
        all_y_train = [all_y_train] if not isinstance(all_y_train, list) else all_y_train
        all_X_val = [all_X_val] if not isinstance(all_X_val, list) else all_X_val
        all_y_val = [all_y_val] if not isinstance(all_y_val, list) else all_y_val

        if self.add_padding:
            # Füge Padding zwischen den DataFrames ein
            X_train = BaseDataClass.insert_padding(all_X_train)
            X_val = BaseDataClass.insert_padding(all_X_val)
            y_train = BaseDataClass.insert_padding(all_y_train)
            y_val = BaseDataClass.insert_padding(all_y_val)
        else:
            # Kombiniere die DataFrames ohne Padding
            X_train = pd.concat(all_X_train).reset_index(drop=True)
            X_val = pd.concat(all_X_val).reset_index(drop=True)
            y_train = pd.concat(all_y_train).reset_index(drop=True)
            y_val = pd.concat(all_y_val).reset_index(drop=True)


        return X_train, X_val, X_test, y_train, y_val, y_test

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def get_documentation(self):
        pass

    def get_test_data_as_pd(self):
        """
        Loads and preprocesses test data for evaluation purposes.
        Applies a rolling mean to the data and adjusts the dataset based on specified past and future values.

        Returns
        -------
        tuple
            A tuple containing the preprocessed test data and base names of the test files:
            - test_datas : list of pd.DataFrame
                Preprocessed test data with rolling mean applied.
            - base_names : list of str
                Base names of the test files without the index part.
        """

        # Getting validation data to right format:
        fulltestdatas = read_fulldata(self.testing_data_paths, self.folder)

        # Apply rolling mean to each DataFrame in the lists
        test_datas = apply_action(fulltestdatas,
                                  lambda data: data[HEADER].rolling(window=self.window_size, min_periods=1).mean())

        if self.past_values + self.future_values != 0:
            test_datas = apply_action(test_datas, lambda target: target.iloc[self.past_values:-self.future_values])

        base_names = []
        for path in self.testing_data_paths:
            file_name = os.path.basename(path)
            name_without_extension = os.path.splitext(file_name)[0]
            # Split the name by underscore and remove the last part (the index)
            base_names.append('_'.join(name_without_extension.split('_')[:-1]))
        return test_datas, base_names

    def load_raw_test_data(self):
        """
        Loads and preprocesses the full dataframe.
        Applies a rolling mean to the data and adjusts the dataset based on specified past and future values.

        Parameters
        ----------
        paths: list of str
            A list of paths to the data files.

        Returns
        -------
        tuple
            A list containing the raw data:
            - datas : list of pd.DataFrame
                Preprocessed test data with rolling mean applied.
        """
        # Getting validation data to right format:
        fulltestdatas = read_fulldata(self.testing_data_paths, self.folder)

        # Apply rolling mean to each DataFrame in the lists
        test_datas = apply_action(fulltestdatas,
                                  lambda data: data.rolling(window=self.window_size, min_periods=1).mean())

        if self.past_values + self.future_values != 0:
            test_datas = apply_action(test_datas, lambda target: target.iloc[self.past_values:-self.future_values])

        return test_datas

# Datenklassen
class DataClass(BaseDataClass):
    def __init__(self, name, folder, training_data_paths, validation_data_paths, testing_data_paths, target_channels = HEADER_y, header = HEADER_x,
                 past_values=0, future_values=0, window_size=1, **kwargs):
        BaseDataClass.__init__(self, name, folder, testing_data_paths, **kwargs)

        self.target_channels = target_channels
        self.header = header

        self.past_values = past_values
        self.future_values = future_values
        if window_size == 0:
            self.window_size = 1
        elif window_size < 0:
            self.window_size = abs(self.window_size)
        else:
            self.window_size = window_size

        self.training_data_paths = training_data_paths
        self.validation_data_paths = validation_data_paths
        self.testing_data_paths = testing_data_paths

        self.use_filter = False
        self.cutoff = 5
        self.filter_order = 4

        self.add_sign_hold = False
        self.add_sign_y = False
        self.columns_to_integrate = []

    def butter_lowpass(self, cutoff, order, nyq_freq=25):
        normal_cutoff = cutoff / nyq_freq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def apply_lowpass_filter(self, data, cutoff, order):
        b, a = self.butter_lowpass(cutoff, order)
        data2 = data.copy()
        for col in data.columns:
            data2[col] = filtfilt(b, a, data2[col])
        return data2

    def check_data_overlap(self):
        """
        Checks if test data is included in training or validation data.

        Returns
        -------
        tuple
            (bool, bool): A tuple indicating whether test data is included in training data and whether test data is included in validation data.
        """
        # Names of test files
        test_files = set(os.path.basename(p) for p in self.testing_data_paths)

        # Names of training files
        training_files = set(os.path.basename(p) for p in self.training_data_paths)

        # Names of validation files
        validation_files = set(os.path.basename(p) for p in self.validation_data_paths)

        # Check for overlaps
        overlap_with_training = not test_files.isdisjoint(training_files)
        overlap_with_validation = not test_files.isdisjoint(validation_files)

        return overlap_with_training, overlap_with_validation

    def add_z_to_data(self, data):
        output = data.copy()
        for header in output.columns:
            if header.startswith('v'):
                output[header.replace('v', 'z')] = sign_hold(output[header].values)

        return output

    def integrate_columns(self, data):
        """
        Integriert die Spalten in data, falls sie in columns_to_integrate vorhanden sind.
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame mit den zu integrierenden Spalten.
        Returns
        -------
        pd.DataFrame
            DataFrame mit den integrierten Spalten.
        """
        for column in self.columns_to_integrate:
            if column in data.columns:
                data[column] = data[column].cumsum()
                print(f'{column} integrated')
        return data

    def load_data_from_path(self, data_paths):
        fulldatas = read_fulldata(data_paths, self.folder)
        datas = apply_action(fulldatas,
                             lambda data: data[self.header].rolling(window=self.window_size, min_periods=1).mean())

        if self.columns_to_integrate:
            datas = apply_action(datas, lambda data: self.integrate_columns(data))

        X = apply_action(datas,
                         lambda data: create_full_ml_vector_optimized(self.past_values, self.future_values, data))
        targets = apply_action(fulldatas, lambda data: data[self.target_channels])
        Y = apply_action(targets, lambda target: target.rolling(window=self.window_size, min_periods=1).mean())

        if self.use_filter:
            X = apply_action(X, lambda data: self.apply_lowpass_filter(data, self.cutoff, self.filter_order))
            Y = apply_action(Y, lambda data: self.apply_lowpass_filter(data, self.cutoff, self.filter_order))

        if self.add_sign_hold:
            X = apply_action(X, lambda data: self.add_z_to_data(data))

        if self.add_sign_y:
            # Add the sign of the previous y value as a new column in x
            for x_df, y_df in zip(X, Y):
                for col in y_df.columns:
                    y_col_sign = f"{col}_sign"
                    x_df[y_col_sign] = y_df[col].shift(1).apply(lambda x: 0 if pd.isna(x) else x).fillna(0)

        if self.past_values + self.future_values != 0:
            Y = apply_action(Y, lambda target: target.iloc[self.past_values:-self.future_values])

        if len(data_paths) <= 1:
            X = pd.concat(X).reset_index(drop=True)
            Y = pd.concat(Y).reset_index(drop=True)

        return X, Y

    def load_data(self):
        """
        Loads and preprocesses data for training, validation, and testing.

        Returns
        -------
        tuple
            X_train, X_val, X_test, y_train, y_val, y_test
            :param
        """
        # Load test data
        X_test, y_test = self.load_data_from_path(self.testing_data_paths)

        # Check for overlaps with assert
        overlap_with_training, overlap_with_validation = self.check_data_overlap()
        assert not overlap_with_training, "Warning: Test data is included in the training data."
        assert not overlap_with_validation, "Warning: Test data is included in the validation data."

        # Load training and validation data
        X_train, y_train = self.load_data_from_path(self.training_data_paths)
        X_val, y_val = self.load_data_from_path(self.validation_data_paths)

        return self.prepare_output(X_train, X_val, X_test, y_train, y_val, y_test)

    def get_documentation(self):
        """
        Gibt eine Dokumentation der Klasse als Dictionary zurück.

        Returns
        -------
        dict
            Ein Dictionary mit den serialisierbaren Attributen der Klasse.
        """
        documentation = {
            "name": self.name,
            "folder": self.folder,
            "training_data_paths": self.training_data_paths,
            "validation_data_paths": self.validation_data_paths,
            "testing_data_paths": self.testing_data_paths,
            "target_channels": self.target_channels,
            'input_features': list(self.header) if hasattr(self, 'header') else None,
            'add_sign_hold': self.add_sign_hold,
            'columns_to_integrate': list(self.columns_to_integrate) if hasattr(self, 'columns_to_integrate') else None
        }
        return documentation

folder_data = '..\\..\\DataSets\\Data'

dataPaths_Test = [  'AL_2007_T4_Gear_Normal_3.csv','AL_2007_T4_Plate_Normal_3.csv', 'S235JR_Gear_Normal_3.csv','S235JR_Plate_Normal_3.csv']

dataPaths_Train = ['S235JR_Plate_Normal_1.csv', 'S235JR_Plate_Normal_2.csv',
                                                    'S235JR_Plate_SF_1.csv', 'S235JR_Plate_Depth_1.csv',
                                                    'S235JR_Plate_SF_2.csv', 'S235JR_Plate_Depth_2.csv',
                                                    'S235JR_Plate_SF_3.csv', 'S235JR_Plate_Depth_3.csv']

dataPaths_Val = ['S235JR_Notch_Normal_1.csv', 'S235JR_Notch_Normal_2.csv', 'S235JR_Notch_Normal_3.csv',
                                              'S235JR_Notch_Depth_1.csv', 'S235JR_Notch_Depth_2.csv', 'S235JR_Notch_Depth_3.csv']



DataClass_ST_Plate_Notch = DataClass('ST_Plate_Notch', folder_data,
                                    dataPaths_Train, dataPaths_Val, dataPaths_Test,
                                             ["curr_x"],
                                     header = HEADER_x)

DataClass_ST_Plate_Notch_noDepth = DataClass('ST_Plate_Notch_noDepth', folder_data,
                                    ['S235JR_Plate_Normal_1.csv', 'S235JR_Plate_Normal_2.csv',
                                                    'S235JR_Plate_SF_1.csv',
                                                    'S235JR_Plate_SF_2.csv', 'S235JR_Plate_Depth_2.csv',
                                                    'S235JR_Plate_SF_3.csv', ],
                                             ['S235JR_Notch_Normal_1.csv', 'S235JR_Notch_Normal_2.csv', 'S235JR_Notch_Normal_3.csv',
                                              'S235JR_Notch_Depth_2.csv'],
                                             dataPaths_Test,
                                             ["curr_x"], header = HEADER_x)

DataClass_ST_Notch_Plate = DataClass('ST_Notch_Plate', folder_data,
                                               ['S235JR_Notch_Normal_1.csv', 'S235JR_Notch_Normal_2.csv',
                                                'S235JR_Notch_Depth_1.csv', 'S235JR_Notch_Depth_2.csv',
                                                'S235JR_Notch_Depth_3.csv'],
                                    ['S235JR_Plate_Normal_1.csv', 'S235JR_Plate_Normal_2.csv',
                                                    'S235JR_Plate_SF_1.csv', 'S235JR_Plate_Depth_1.csv',
                                                    'S235JR_Plate_SF_2.csv', 'S235JR_Plate_Depth_2.csv',
                                                    'S235JR_Plate_SF_3.csv', 'S235JR_Plate_Depth_3.csv'],

                                             [  'AL_2007_T4_Gear_Normal_3.csv','AL_2007_T4_Notch_Normal_3.csv', 'S235JR_Gear_Normal_3.csv','S235JR_Notch_Normal_3.csv'],
                                             ["curr_x"], header = HEADER_x)

DataClass_ST_Plate_Notch_Mes = DataClass('ST_Plate_Notch_Mesurments', folder_data,
                                    dataPaths_Train, dataPaths_Val, dataPaths_Test,
                                             ["curr_x"], header = ["v_sp", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "a_sp", "f_x", "f_y", "f_z"])

def sign_hold(v, eps = 1e-1):
    # Initialisierung des Arrays z mit Nullen
    z = np.zeros(len(v))

    # Initialisierung des FiFo h mit Länge 5 und Initialwerten 0
    h = deque([0, 0, 0, 0, 0], maxlen=5)

    # Berechnung von z
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])

        if i >= 4:  # Da wir ab dem 5. Element starten wollen
            # Berechne zi als Vorzeichen der Summe
            z[i] = np.sign(sum(h))

    return z

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
        # Load .pkl files and convert the data to a DataFrame
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            data = data.T
            df = pd.DataFrame(data)
            if 'Air' in file_path:
                df.columns = ['pos_x', 'pos_y', 'pos_z', 'pos_sp', 'curr_x', 'curr_y', 'curr_z', 'curr_sp']
            elif 'Mat' in file_path:
                df.columns = ['pos_x', 'pos_y', 'pos_z', 'pos_sp',
                              'f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim', 'materialremoved_sim',
                              'curr_x', 'curr_y', 'curr_z', 'curr_sp']
            else:
                raise ValueError("Invalid file name. File name must include Mat or Air")
            for pos in ['pos_x', 'pos_y', 'pos_z', 'pos_sp']:
                axis = pos.replace('pos_', '')
                df[f'v_{axis}'] = df[pos].diff()
                df[f'a_{axis}'] = df[f'v_{axis}'].diff()
            df.dropna(inplace=True)
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