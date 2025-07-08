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
from collections import Counter, deque

import jax
import jax.numpy as jnp
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from numpy.f2py.auxfuncs import throw_error
from numpy.ma.core import anomalies
from scipy import stats
import re

import pandas as pd
import numpy as np
import pickle

from abc import ABC, abstractmethod

from sympy import false

# konstanten
WINDOWSIZE = 1

HEADER = ["pos_x", "pos_y", "pos_z", "v_sp", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "a_sp", "curr_x", "curr_y", "curr_z", "curr_sp", "f_x_sim", "f_y_sim", "f_z_sim", "f_sp_sim", "materialremoved_sim"]
HEADER_x = ["v_sp", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "a_sp", "f_x_sim", "f_y_sim", "f_z_sim", "f_sp_sim", "materialremoved_sim"]
HEADER_v = ["v_sp", "v_x", "v_y", "v_z"]
HEADER_y = ["curr_x", "curr_y", "curr_z", "curr_sp"]

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

class BaseDataClass(ABC):
    @abstractmethod
    def __init__(self, name, folder, testing_data_paths, target_channels=HEADER_y, do_preprocessing=True, n=12, past_values=2, future_values=2, window_size=1, keep_separate=False):
        self.name = name
        self.folder = folder
        self.target_channels = target_channels
        self.do_preprocessing = do_preprocessing
        self.n = n
        self.past_values = past_values
        self.future_values = future_values
        if window_size == 0:
            self.window_size = 1
        elif window_size < 0:
            self.window_size = abs(self.window_size)
        else:
            self.window_size = window_size
        self.keep_separate = keep_separate
        self.testing_data_paths = testing_data_paths

    def prepare_output(self, all_X_train, all_X_val, X_test, all_y_train, all_y_val, y_test):
        """
        Returns
        -------
        tuple
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        if type(all_X_train) is not list:
            all_X_train = [all_X_train]
            all_y_train = [all_y_train]
        if type(all_X_val) is not list:
            all_X_val = [all_X_val]
            all_y_val = [all_y_val]
        if self.keep_separate:
            return all_X_train, all_X_val, X_test, all_y_train, all_y_val, y_test
        else:
            X_train = pd.concat(all_X_train).reset_index(drop=True)
            y_train = pd.concat(all_y_train).reset_index(drop=True)
            X_val = pd.concat(all_X_val).reset_index(drop=True)
            y_val = pd.concat(all_y_val).reset_index(drop=True)

            return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocessing(self, X, y):
        """
        Preprocesses the data.

        Parameters
        ----------
        X : DataFrame or list of DataFrames
            Input data.
        y : DataFrame or list of DataFrames
            Target data.

        Returns
        -------
        tuple
            Preprocessed X and y.
        """
        if isinstance(X, list):
            # Initialize lists to store preprocessed X and y
            X_preprocessed = []
            y_preprocessed = []

            # Iterate over each pair of X and y, preprocess them, and store the results
            for x, y in zip(X, y):
                x_cleaned, y_cleaned = self._preprocess_single(x, y)
                X_preprocessed.append(x_cleaned)
                y_preprocessed.append(y_cleaned)

            return X_preprocessed, y_preprocessed
        else:
            return self._preprocess_single(X, y)

    def _preprocess_single(self, X, y):
        """
        Remove outliers from the input data and adjust the corresponding y values.

        Parameters:
        X (DataFrame or Series): The input data.
        y (Series or DataFrame): The corresponding target values.
        n (int): The number of standard deviations to consider for outlier removal. Default is 12.

        Returns:
        Tuple: The cleaned input data and the adjusted target values.
        """
        y.index = X.index

        if isinstance(y, pd.Series):
            # Calculate the mean and standard deviation for the target values
            mean = y.mean()
            std = y.std()

            # Identify outliers in the target values
            outliers = np.abs(y - mean) > self.n * std
        elif isinstance(y, pd.DataFrame):
            # Calculate the mean and standard deviation for each feature
            mean = y.mean(axis=0)
            std = y.std(axis=0)

            # Identify outliers
            outliers = np.abs(y - mean) > self.n * std
            outliers = outliers.any(axis=1)
        else:
            raise ValueError("y must be a pandas Series or DataFrame")

        # Remove rows that contain outliers
        x_cleaned = X[~outliers]
        y_cleaned = y[~outliers]

        return x_cleaned, y_cleaned

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
    def __init__(self, name, folder, training_data_paths, validation_data_paths, testing_data_paths, target_channels = HEADER_y, do_preprocessing=True, n=12, past_values=2, future_values=2, window_size=1, keep_separate=False):
        self.name = name
        self.folder = folder
        self.target_channels = target_channels
        self.do_preprocessing = do_preprocessing
        self.n = n
        self.past_values = past_values
        self.future_values = future_values
        if window_size == 0:
            self.window_size = 1
        elif window_size < 0:
            self.window_size = abs(self.window_size)
        else:
            self.window_size = window_size
        self.keep_separate = keep_separate
        self.training_data_paths = training_data_paths
        self.validation_data_paths = validation_data_paths
        self.testing_data_paths = testing_data_paths

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

    def load_data_from_path(self, data_paths):
        """
        Loads and preprocesses data from given paths.

        Parameters
        ----------
        data_paths : list
            List of paths to the data files.

        Returns
        -------
        tuple
            X, Y: Preprocessed data and targets.
        """

        # Load data
        fulldatas = read_fulldata(data_paths, self.folder)
        datas = apply_action(fulldatas, lambda data: data[HEADER_x].rolling(window=self.window_size, min_periods=1).mean())
        X = apply_action(datas, lambda data: create_full_ml_vector_optimized(self.past_values, self.future_values, data))
        targets = apply_action(fulldatas, lambda data: data[self.target_channels])
        Y = apply_action(targets, lambda target: target.rolling(window=self.window_size, min_periods=1).mean())
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

        if self.do_preprocessing:
            X_train, y_train = self.preprocessing(X_train, y_train)
            X_val, y_val = self.preprocessing(X_val, y_val)

        return self.prepare_output(X_train, X_val, X_test, y_train, y_val, y_test)

    def get_documentation(self):
        documentation = {
            "name": self.name,
            "folder": self.folder,
            "training_data_paths": self.training_data_paths,
            "validation_data_paths": self.validation_data_paths,
            "testing_data_paths": self.testing_data_paths,
            "target_channels": self.target_channels,
        }
        return documentation

class DataClassSingleAxis(BaseDataClass):
    def __init__(self, name, folder, training_data_paths, validation_data_paths, testing_data_paths, target_channels = HEADER_y, axis='x', do_preprocessing=True, n=12, past_values=2, future_values=2, window_size=1, keep_separate=False):
        self.name = name
        self.folder = folder
        self.target_channels = target_channels
        self.do_preprocessing = do_preprocessing
        self.n = n
        self.past_values = past_values
        self.future_values = future_values
        if window_size == 0:
            self.window_size = 1
        elif window_size < 0:
            self.window_size = abs(self.window_size)
        else:
            self.window_size = window_size
        self.keep_separate = keep_separate
        self.axis = axis
        self.training_data_paths = training_data_paths
        self.validation_data_paths = validation_data_paths
        self.testing_data_paths = testing_data_paths

    def create_full_ml_vector_optimized(self, past_values, future_values, channels_in: pd.DataFrame) -> np.array:
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
        df_filtered = channels_in.filter(regex=self.axis)
        df_filtered['materialremoved_sim'] = channels_in['materialremoved_sim']
        n = len(df_filtered)

        # Remove past_values from the beginning and future_values from the end
        # channels_in = channels_in.iloc[past_values:n - future_values]

        full_vector = pd.DataFrame(index=range(n - (past_values + future_values)))

        # Determine the maximum length of the numbers in the column names
        max_digits = len(str(len(df_filtered.columns)))

        for i in range(past_values + future_values + 1):
            if i < past_values:
                shifted = df_filtered.shift(-i)
                shifted.columns = [f'{str(col).zfill(max_digits)}_0_past_{i}' for col in df_filtered.columns]
            elif i == past_values:
                shifted = df_filtered.shift(-past_values)
                shifted.columns = [f'{str(col).zfill(max_digits)}_1_current' for col in df_filtered.columns]
            else:
                shifted = df_filtered.shift(-i)
                shifted.columns = [f'{str(col).zfill(max_digits)}_2_future_{i - past_values - 1}' for col in
                                   df_filtered.columns]

            full_vector = pd.concat([full_vector, shifted], axis=1).dropna()

        # Sort column names
        sorted_columns = sorted(full_vector.columns)

        # Create DataFrame with sorted column names
        full_vector = full_vector[sorted_columns]
        return full_vector

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

    def load_data_from_path(self, data_paths):
        """
        Loads and preprocesses data from given paths.

        Parameters
        ----------
        data_paths : list
            List of paths to the data files.

        Returns
        -------
        tuple
            X, Y: Preprocessed data and targets.
        """
        # Load data
        fulldatas = read_fulldata(data_paths, self.folder)
        datas = apply_action(fulldatas, lambda data: data[HEADER_x].rolling(window=self.window_size, min_periods=1).mean())
        X = apply_action(datas, lambda data: self.create_full_ml_vector_optimized(self.past_values, self.future_values, data))
        targets = apply_action(fulldatas, lambda data: data[self.target_channels])
        Y = apply_action(targets, lambda target: target.rolling(window=self.window_size, min_periods=1).mean())
        if self.past_values + self.future_values != 0:
            Y = apply_action(Y, lambda target: target.iloc[self.past_values:-self.future_values])

        if len(data_paths) <= 1:
            X = pd.concat(X).reset_index(drop=True)
            Y = pd.concat(Y).reset_index(drop=True)

        return X, Y

    def load_data(self):
        """
        Loads and preprocesses data for training, validation, and testing.

        Parameters
        ----------
        self.past_values : int, optional
            The number of past values to consider. The default is 2.

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

        if self.do_preprocessing:
            X_train, y_train = self.preprocessing(X_train, y_train)
            X_val, y_val = self.preprocessing(X_val, y_val)

        return self.prepare_output(X_train, X_val, X_test, y_train, y_val, y_test)

    def get_documentation(self):
        documentation = {
            "name": self.name,
            "folder": self.folder,
            "training_data_paths": self.training_data_paths,
            "validation_data_paths": self.validation_data_paths,
            "testing_data_paths": self.testing_data_paths,
            "target_channels": self.target_channels,
        }
        return documentation

# Enable JIT compilation for all functions
@partial(jax.jit, static_argnums=(0,))
def rk4_step(f, y, t, dt, *args):
    """Single Runge-Kutta 4th order step - JIT compiled"""
    k1 = f(y, t, *args)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt, *args)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt, *args)
    k4 = f(y + dt * k3, t + dt, *args)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

@partial(jax.jit, static_argnums=(0,))
def rk4_scan_integrate(f, y0, t_array, *static_args):
    def scan_fn(y, t_curr):
        dt = t_array[1] - t_array[0]
        y_next = rk4_step(f, y, t_curr, dt, *static_args)
        return y_next, y_next

    _, y_values = jax.lax.scan(scan_fn, y0, t_array[1:])
    return jnp.concatenate([jnp.array([y0]), y_values])

class DataclassCombinedTrainVal(BaseDataClass):
    def __init__(self, name, folder, training_validation_datas, testing_data_paths, target_channels=HEADER_y,
                 do_preprocessing=True, n=12, past_values=2, future_values=2, window_size=1, keep_separate=False, N = 3,
                 header=HEADER_x):
        self.name = name
        self.folder = folder
        self.target_channels = target_channels
        self.do_preprocessing = do_preprocessing
        self.n = n
        self.past_values = past_values
        self.future_values = future_values
        if window_size == 0:
            self.window_size = 1
        elif window_size < 0:
            self.window_size = abs(self.window_size)
        else:
            self.window_size = window_size
        self.keep_separate = keep_separate
        self.N = N
        self.training_validation_datas = training_validation_datas
        self.testing_data_paths = testing_data_paths
        self.add_sign_hold = False
        self.header = header

    def load_data(self):
        """
        Loads and preprocesses data for training, validation, and testing.

        Returns
        -------
        tuple
            X_train, X_val, X_test, y_train, y_val, y_test
        """

        # Load test data
        fulltestdatas = read_fulldata(self.testing_data_paths, self.folder)
        test_datas = apply_action(fulltestdatas,
                                  lambda data: data[self.header].rolling(window=self.window_size, min_periods=1).mean())

        header_v = []
        for column in self.header:
            if column.startswith('v_'):
                header_v.append(column)

        def integrand_x0(x0, t_curr, v_data):
            idx = jnp.argmin(jnp.abs(t_curr - jnp.arange(len(v_data))))
            v_interp = v_data[idx]
            return v_interp

        for column in self.header:
            if column.startswith('CONT_DEV_'):
                for data in test_datas:
                    cont_dev = jnp.array(data[column])
                    # Define the time array
                    t = jnp.linspace(0, len(cont_dev) - 1, len(cont_dev))

                    # Integrate cont_dev_x using RK4
                    integrated_cont_dev_x = rk4_scan_integrate(partial(integrand_x0, v_data=cont_dev),
                                                               jnp.array([cont_dev[0]]), t)

                    data[column.replace('CONT_DEV_', 'dev_int_')] = integrated_cont_dev_x

        if self.add_sign_hold:
            for data in test_datas:
                for header in header_v:
                    data[header.replace('v', 'z')] = sign_hold(data[header].values)

        X_test = apply_action(test_datas,
                              lambda data: create_full_ml_vector_optimized(self.past_values, self.future_values, data))
        test_targets = apply_action(fulltestdatas, lambda data: data[self.target_channels])
        y_test = apply_action(test_targets, lambda target: target.rolling(window=self.window_size, min_periods=1).mean())
        if self.past_values + self.future_values != 0:
            y_test = apply_action(y_test, lambda target: target.iloc[self.past_values:-self.future_values])

        # Namen der Testdateien zum Ausschluss
        test_files = set(os.path.basename(p) for p in self.testing_data_paths)

        # Trainings- und Validierungsdaten vorbereiten
        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []

        toggle = True  # Start mit gerade = Train

        for name in self.training_validation_datas:
            pattern = f"{name}_*.csv"
            files = glob.glob(os.path.join(self.folder, pattern))
            files = [f for f in files if os.path.basename(f) not in test_files]

            file_datas = read_fulldata(files, self.folder)
            file_datas_x = apply_action(file_datas, lambda data: data[self.header].rolling(window=self.window_size,
                                                                                        min_periods=1).mean())
            file_datas_y = apply_action(file_datas,
                                        lambda data: data[self.target_channels].rolling(window=self.window_size,
                                                                                               min_periods=1).mean())
            if self.add_sign_hold:
                for data in file_datas_x:
                    for header in header_v:
                        data[header.replace('v', 'z')] = sign_hold(data[header].values)

            for column in self.header:
                if column.startswith('CONT_DEV_'):
                    for data in file_datas_x:
                        cont_dev = jnp.array(data[column])
                        # Define the time array
                        t = jnp.linspace(0, len(cont_dev) - 1, len(cont_dev))

                        # Integrate cont_dev_x using RK4
                        integrated_cont_dev_x = rk4_scan_integrate(partial(integrand_x0, v_data=cont_dev),
                                                                   jnp.array([cont_dev[0]]), t)

                        data[column.replace('CONT_DEV_', 'dev_int_')] = integrated_cont_dev_x

            X_files = apply_action(file_datas_x,
                                   lambda data: create_full_ml_vector_optimized(self.past_values, self.future_values,
                                                                                data))
            y_files = apply_action(file_datas_y, lambda target: target.iloc[
                                                                self.past_values:-self.future_values] if self.past_values + self.future_values != 0 else target)

            for X_df, y_df in zip(X_files, y_files):
                X_split = np.array_split(X_df, self.N)
                y_split = np.array_split(y_df, self.N)

                if toggle:
                    train_indices = [i for i in range(self.N) if i % 2 == 0]
                    val_indices = [i for i in range(self.N) if i % 2 != 0]
                else:
                    train_indices = [i for i in range(self.N) if i % 2 != 0]
                    val_indices = [i for i in range(self.N) if i % 2 == 0]

                X_train_parts = [X_split[i].reset_index(drop=True) for i in train_indices]
                y_train_parts = [y_split[i].reset_index(drop=True) for i in train_indices]
                X_val_parts = [X_split[i].reset_index(drop=True) for i in val_indices]
                y_val_parts = [y_split[i].reset_index(drop=True) for i in val_indices]

                if self.do_preprocessing:
                    X_train_parts, y_train_parts = zip(*[self.preprocessing(X, y) for X, y in zip(X_train_parts, y_train_parts)])
                    X_val_parts, y_val_parts = zip(*[self.preprocessing(X, y) for X, y in zip(X_val_parts, y_val_parts)])

                all_X_train.extend(X_train_parts)
                all_y_train.extend(y_train_parts)
                all_X_val.extend(X_val_parts)
                all_y_val.extend(y_val_parts)

            toggle = not toggle  # Umschalten für nächste Datei

        if len(X_test) <= 1:
            X_test = pd.concat(X_test).reset_index(drop=True)
            y_test = pd.concat(y_test).reset_index(drop=True)

        return self.prepare_output(all_X_train, all_X_val, X_test, all_y_train, all_y_val, y_test)

    def get_documentation(self):
        documentation = {
            "name": self.name,
            "folder": self.folder,
            "training_validation_data": self.training_validation_datas,
            "testing_data_paths": self.testing_data_paths,
            "target_channels": self.target_channels,
            "add_sign_hold": self.add_sign_hold,
            "Header": self.header,
        }
        return documentation

class DataclassCombinedTrainValSingleAxis(BaseDataClass):
    def __init__(self, name, folder, training_validation_datas, testing_data_paths, target_channels=HEADER_y, do_preprocessing=True, n=12, past_values=2, future_values=2, window_size=1, keep_separate=False, N = 3, axis = 'x', load_all_geometrie_variations=True):
        self.name = name+'_SingleAxis'
        self.folder = folder
        self.target_channels = target_channels
        self.do_preprocessing = do_preprocessing
        self.n = n
        self.past_values = past_values
        self.future_values = future_values
        if window_size == 0:
            self.window_size = 1
        elif window_size < 0:
            self.window_size = abs(self.window_size)
        else:
            self.window_size = window_size
        self.keep_separate = keep_separate
        self.N = N
        self.training_validation_datas = training_validation_datas
        self.testing_data_paths = testing_data_paths
        self.axis = axis

    def create_full_ml_vector_optimized(self, past_values, future_values, channels_in: pd.DataFrame) -> np.array:
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
        df_filtered = channels_in.filter(regex=self.axis)
        df_filtered['materialremoved_sim'] = channels_in['materialremoved_sim']
        n = len(df_filtered)

        # Remove past_values from the beginning and future_values from the end
        # channels_in = channels_in.iloc[past_values:n - future_values]

        full_vector = pd.DataFrame(index=range(n - (past_values + future_values)))

        # Determine the maximum length of the numbers in the column names
        max_digits = len(str(len(df_filtered.columns)))

        for i in range(past_values + future_values + 1):
            if i < past_values:
                shifted = df_filtered.shift(-i)
                shifted.columns = [f'{str(col).zfill(max_digits)}_0_past_{i}' for col in df_filtered.columns]
            elif i == past_values:
                shifted = df_filtered.shift(-past_values)
                shifted.columns = [f'{str(col).zfill(max_digits)}_1_current' for col in df_filtered.columns]
            else:
                shifted = df_filtered.shift(-i)
                shifted.columns = [f'{str(col).zfill(max_digits)}_2_future_{i - past_values - 1}' for col in
                                   df_filtered.columns]

            full_vector = pd.concat([full_vector, shifted], axis=1).dropna()

        # Sort column names
        sorted_columns = sorted(full_vector.columns)

        # Create DataFrame with sorted column names
        full_vector = full_vector[sorted_columns]
        return full_vector

    def load_data(self):
        """
        Loads and preprocesses data for training, validation, and testing.

        Returns
        -------
        tuple
            X_train, X_val, X_test, y_train, y_val, y_test
        """

        # Load test data
        fulltestdatas = read_fulldata(self.testing_data_paths, self.folder)
        test_datas = apply_action(fulltestdatas,
                                  lambda data: data[HEADER_x].rolling(window=self.window_size, min_periods=1).mean())
        X_test = apply_action(test_datas,
                              lambda data: self.create_full_ml_vector_optimized(self.past_values, self.future_values, data))
        test_targets = apply_action(fulltestdatas, lambda data: data[self.target_channels])
        y_test = apply_action(test_targets, lambda target: target.rolling(window=self.window_size, min_periods=1).mean())
        if self.past_values + self.future_values != 0:
            y_test = apply_action(y_test, lambda target: target.iloc[self.past_values:-self.future_values])

        # Namen der Testdateien zum Ausschluss
        test_files = set(os.path.basename(p) for p in self.testing_data_paths)

        # Trainings- und Validierungsdaten vorbereiten
        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []

        toggle = True  # Start mit gerade = Train

        for name in self.training_validation_datas:
            pattern = f"{name}_*.csv"
            files = glob.glob(os.path.join(self.folder, pattern))
            files = [f for f in files if os.path.basename(f) not in test_files]

            file_datas = read_fulldata(files, self.folder)
            file_datas_x = apply_action(file_datas, lambda data: data[HEADER_x].rolling(window=self.window_size,
                                                                                        min_periods=1).mean())
            file_datas_y = apply_action(file_datas,
                                        lambda data: data[self.target_channels].rolling(window=self.window_size,
                                                                                               min_periods=1).mean())

            X_files = apply_action(file_datas_x,
                                   lambda data: self.create_full_ml_vector_optimized(self.past_values, self.future_values,
                                                                                data))
            y_files = apply_action(file_datas_y, lambda target: target.iloc[
                                                                self.past_values:-self.future_values] if self.past_values + self.future_values != 0 else target)

            for X_df, y_df in zip(X_files, y_files):
                X_split = np.array_split(X_df, self.N)
                y_split = np.array_split(y_df, self.N)

                if toggle:
                    train_indices = [i for i in range(self.N) if i % 2 == 0]
                    val_indices = [i for i in range(self.N) if i % 2 != 0]
                else:
                    train_indices = [i for i in range(self.N) if i % 2 != 0]
                    val_indices = [i for i in range(self.N) if i % 2 == 0]

                X_train_parts = [X_split[i].reset_index(drop=True) for i in train_indices]
                y_train_parts = [y_split[i].reset_index(drop=True) for i in train_indices]
                X_val_parts = [X_split[i].reset_index(drop=True) for i in val_indices]
                y_val_parts = [y_split[i].reset_index(drop=True) for i in val_indices]

                if self.do_preprocessing:
                    X_train_parts, y_train_parts = zip(*[self.preprocessing(X, y) for X, y in zip(X_train_parts, y_train_parts)])
                    X_val_parts, y_val_parts = zip(*[self.preprocessing(X, y) for X, y in zip(X_val_parts, y_val_parts)])

                all_X_train.extend(X_train_parts)
                all_y_train.extend(y_train_parts)
                all_X_val.extend(X_val_parts)
                all_y_val.extend(y_val_parts)

            toggle = not toggle  # Umschalten für nächste Datei

        if len(X_test) <= 1:
            X_test = pd.concat(X_test).reset_index(drop=True)
            y_test = pd.concat(y_test).reset_index(drop=True)

        return self.prepare_output(all_X_train, all_X_val, X_test, all_y_train, all_y_val, y_test)

    def get_documentation(self):
        documentation = {
            "name": self.name,
            "folder": self.folder,
            "training_validation_data": self.training_validation_datas,
            "testing_data_paths": self.testing_data_paths,
            "target_channels": self.target_channels,
        }
        return documentation

folder_data = '..\\..\\DataSets\DataFiltered'
Al_Al_Gear_Plate = DataclassCombinedTrainVal('Al_Al_Gear_Plate', folder_data,
                                             ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                             ['AL_2007_T4_Plate_Normal_3.csv'],
                                             ["curr_x"])
Al_St_Gear_Gear = DataclassCombinedTrainVal('Al_St_Gear_Gear', folder_data,
                                            ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                            ['S235JR_Gear_Normal_3.csv'],
                                            ["curr_x"])
Al_St_Gear_Plate = DataclassCombinedTrainVal('Al_St_Gear_Plate', folder_data,
                                             ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                             ['S235JR_Plate_Normal_3.csv'],
                                             ["curr_x"])
dataSets_list_Gear = [Al_Al_Gear_Plate,Al_St_Gear_Gear,Al_St_Gear_Plate]

Al_Al_Plate_Gear = DataclassCombinedTrainVal('Al_Al_Plate_Gear', folder_data,
                                             ['AL_2007_T4_Plate', 'AL_2007_T4_Plate_Depth', 'AL_2007_T4_Plate_SF'],
                                             ['AL_2007_T4_Gear_Normal_3.csv'],
                                             ["curr_x"])
Al_St_Plate_Plate = DataclassCombinedTrainVal('Al_St_Plate_Plate', folder_data,
                                              ['AL_2007_T4_Plate', 'AL_2007_T4_Plate_Depth', 'AL_2007_T4_Plate_SF'],
                                              ['S235JR_Plate_Normal_3.csv'],
                                              ["curr_x"])
Al_St_Plate_Gear = DataclassCombinedTrainVal('Al_St_Plate_Gear', folder_data,
                                             ['AL_2007_T4_Plate', 'AL_2007_T4_Plate_Depth', 'AL_2007_T4_Plate_SF'],
                                             ['S235JR_Gear_Normal_3.csv'],
                                             ["curr_x"])
dataSets_list_Plate = [Al_Al_Plate_Gear,Al_St_Plate_Plate,Al_St_Plate_Gear]

dataPaths_Test = [  'AL_2007_T4_Gear_Normal_3.csv','AL_2007_T4_Plate_Normal_3.csv', 'S235JR_Gear_Normal_3.csv','S235JR_Plate_Normal_3.csv']

dataPaths_Val_KL = ['Kühlgrill_Mat_S2800_1.csv', 'Kühlgrill_Mat_S3800_1.csv', 'Kühlgrill_Mat_S4700_1.csv', 'Laufrad_Durchlauf_1_1.csv', 'Laufrad_Durchlauf_2_1.csv',
                    'Kühlgrill_Mat_S2800_2.csv', 'Kühlgrill_Mat_S3800_2.csv','Kühlgrill_Mat_S4700_1.csv', 'Laufrad_Durchlauf_1_2.csv',  'Laufrad_Durchlauf_2_2.csv',
                    'Kühlgrill_Mat_S2800_3.csv', 'Kühlgrill_Mat_S3800_3.csv','Kühlgrill_Mat_S4700_1.csv', 'Laufrad_Durchlauf_1_3.csv', 'Laufrad_Durchlauf_2_3.csv']

dataPaths_Train_Gear = ['AL_2007_T4_Gear_SF_1.csv', 'AL_2007_T4_Gear_Depth_1.csv', 'AL_2007_T4_Gear_Normal_1.csv',
                    'AL_2007_T4_Gear_SF_2.csv', 'AL_2007_T4_Gear_Depth_2.csv','AL_2007_T4_Gear_Normal_2.csv', ]

dataPaths_Train_Gear_sort = ['AL_2007_T4_Gear_SF_1.csv', 'AL_2007_T4_Gear_Depth_1.csv', 'AL_2007_T4_Gear_Normal_1.csv']

dataPaths_Val_Gear = ['AL_2007_T4_Gear_SF_2.csv', 'AL_2007_T4_Gear_Depth_2.csv', 'AL_2007_T4_Gear_Normal_2.csv', ]

dataPaths_Train_Plate = ['AL_2007_T4_Plate_SF_1.csv', 'AL_2007_T4_Plate_Depth_1.csv', 'AL_2007_T4_Plate_Normal_1.csv',
                    'AL_2007_T4_Plate_SF_2.csv', 'AL_2007_T4_Plate_Depth_2.csv','AL_2007_T4_Plate_Normal_2.csv', ]

dataPaths_Train_Plate_sort = ['AL_2007_T4_Plate_SF_1.csv', 'AL_2007_T4_Plate_Depth_1.csv', 'AL_2007_T4_Plate_Normal_1.csv']

dataPaths_Val_Plate = ['AL_2007_T4_Plate_SF_2.csv', 'AL_2007_T4_Plate_Depth_2.csv', 'AL_2007_T4_Plate_Normal_2.csv', ]

Combined_Gear = DataClass('Gear', folder_data,
                          dataPaths_Train_Gear,
                          dataPaths_Val_KL,
                          dataPaths_Test,
                          ["curr_x"], )
Combined_Gear_2 = DataClass('Gear_2', folder_data,
                            dataPaths_Train_Gear_sort,
                            dataPaths_Val_Gear,
                            dataPaths_Test,
                            ["curr_x"], )
Combined_Gear_Single = DataClassSingleAxis('Gear_Single_Axis', folder_data,
                                           dataPaths_Train_Gear,
                                           dataPaths_Val_KL,
                                           dataPaths_Test,
                                           ["curr_x"], )

Combined_Gear_Single_2 = DataClassSingleAxis('Gear_Single_Axis_2', folder_data,
                                             dataPaths_Train_Gear_sort,
                                             dataPaths_Val_Gear,
                                             dataPaths_Test,
                                             ["curr_x"], )

Combined_Gear_TrainVal = DataclassCombinedTrainVal('Gear_TrainVal', folder_data,
                                                   ['AL_2007_T4_Gear', 'AL_2007_T4_Gear_Depth', 'AL_2007_T4_Gear_SF'],
                                                   dataPaths_Test,
                                                   ["curr_x"], )
Combined_Plate_simple = DataClass('Plate', folder_data,
                          ['AL_2007_T4_Plate_Normal_3.csv'],
                          ['AL_2007_T4_Plate_Depth_2.csv'],
                          [  'AL_2007_T4_Gear_Normal_3.csv','AL_2007_T4_Plate_Normal_2.csv', 'S235JR_Gear_Normal_3.csv','S235JR_Plate_Normal_3.csv'],
                          ["curr_x"], keep_separate= True)
Combined_Plate = DataClass('Plate', folder_data,
                          dataPaths_Train_Plate,
                          dataPaths_Val_KL,
                          dataPaths_Test,
                          ["curr_x"], )

dataPaths_Test_Extended = [ 'AL_2007_T4_Gear_Normal_3.csv','AL_2007_T4_Plate_Normal_3.csv',
                            'AL_2007_T4_Plate_SF_2.csv', 'AL_2007_T4_Plate_Depth_2.csv',
                            'S235JR_Gear_Normal_3.csv','S235JR_Plate_Normal_3.csv']

PPhys = DataClass('PPhys', folder_data,
                        ['AL_2007_T4_Plate_Normal_2.csv'], #
                          ['AL_2007_T4_Plate_Normal_1.csv'], #
                          dataPaths_Test_Extended,
                          ["curr_x"], )

Combined_PPhys_SF = DataClass('PPhys_SF', folder_data,
                        ['AL_2007_T4_Plate_Normal_2.csv', 'AL_2007_T4_Plate_SF_2.csv'], #
                          ['AL_2007_T4_Plate_SF_1.csv', 'AL_2007_T4_Plate_Normal_1.csv'], #
                          dataPaths_Test_Extended,
                          ["curr_x"], )

Combined_PPhys_Depth = DataClass('PPhys_Depth', folder_data,
                        ['AL_2007_T4_Plate_Normal_2.csv', 'AL_2007_T4_Plate_Depth_2.csv'], #
                          ['AL_2007_T4_Plate_Normal_1.csv', 'AL_2007_T4_Plate_Depth_1.csv'], #,  'AL_2007_T4_Gear_Depth_2.csv'
                          dataPaths_Test_Extended,
                          ["curr_x"], )

Combined_PPhys = DataClass('Combined_PPhys', folder_data,
                        ['AL_2007_T4_Plate_Normal_2.csv', 'AL_2007_T4_Plate_SF_2.csv', 'AL_2007_T4_Plate_Depth_2.csv'], #
                          ['AL_2007_T4_Plate_SF_1.csv', 'AL_2007_T4_Plate_Normal_1.csv',  'AL_2007_T4_Plate_Depth_1.csv'], #
                          dataPaths_Test_Extended,
                          ["curr_x"], )

Combined_PPhys_St = DataClass('Combined_PPhys_St', folder_data,
                        ['S235JR_Plate_Normal_2.csv', 'S235JR_Plate_SF_2.csv', 'S235JR_Plate_Depth_2.csv'], #
                          ['S235JR_Plate_SF_1.csv', 'S235JR_Plate_Normal_1.csv',  'S235JR_Plate_Depth_1.csv'], #
                          dataPaths_Test_Extended,
                          ["curr_x"], )

Combined_Plate_2 = DataClass('Plate_2', folder_data,
                            dataPaths_Train_Plate_sort,
                            dataPaths_Val_Plate,
                            dataPaths_Test,
                            ["curr_x"], )
Combined_Plate_Single = DataClassSingleAxis('Plate_Single_Axis', folder_data,
                                           dataPaths_Train_Plate,
                                           dataPaths_Val_KL,
                                           dataPaths_Test,
                                           ["curr_x"], )

Combined_Plate_Single_2 = DataClassSingleAxis('Plate_Single_Axis_2', folder_data,
                                             dataPaths_Train_Plate_sort,
                                             dataPaths_Val_Plate,
                                             dataPaths_Test,
                                             ["curr_x"], )

Combined_Plate_TrainVal = DataclassCombinedTrainVal('Plate_TrainVal', folder_data,
                                                    ['AL_2007_T4_Plate', 'AL_2007_T4_Plate_Depth', 'AL_2007_T4_Plate_SF'],
                                                    dataPaths_Test,
                                                    ["curr_x"], )

Combined_Plate_AlSt_TrainVal = DataclassCombinedTrainVal('Plate_AlSt_TrainVal', folder_data,
                                                    ['AL_2007_T4_Plate', 'AL_2007_T4_Plate_Depth', 'AL_2007_T4_Plate_SF',
                                                     'S235JR_T4_Plate', 'S235JR_T4_Plate_Depth', 'S235JR_Plate_SF'],
                                                    dataPaths_Test,
                                                    ["curr_x"], )
Combined_Plate_St_TrainVal = DataclassCombinedTrainVal('Plate_St_TrainVal', folder_data,
                                                    ['S235JR_T4_Plate', 'S235JR_T4_Plate_Depth', 'S235JR_Plate_SF'],
                                                    dataPaths_Test,
                                                    ["curr_x"], )

Combined_Plate_TrainVal_Single = DataclassCombinedTrainValSingleAxis('Plate_TrainVal', folder_data,
                                                           ['AL_2007_T4_Plate', 'AL_2007_T4_Plate_Depth', 'AL_2007_T4_Plate_SF'],
                                                           dataPaths_Test_Extended,
                                                           ["curr_x"], )

Combined_PK_TrainVal = DataclassCombinedTrainVal('PK_TrainVal', folder_data,
                                                  [   'AL_2007_T4_Plate', 'AL_2007_T4_Plate_Depth', 'AL_2007_T4_Plate_SF',
                                                        'Kühlgrill_Mat_S2800', 'Kühlgrill_Mat_S3800', 'Kühlgrill_Mat_S4700',
                                                        ], # 'Laufrad_Durchlauf_1', 'Laufrad_Durchlauf_2'
                                                  dataPaths_Test_Extended,
                                                  ["curr_x"], )
Combined_Plate_reduced_TrainVal = DataclassCombinedTrainVal('Plate_reduced_TrainVal', folder_data,
                                                  ['AL_2007_T4_Plate_SF', 'AL_2007_T4_Plate_Depth'
                                                        ], # 'Laufrad_Durchlauf_1', 'Laufrad_Durchlauf_2'
                                                  dataPaths_Test,
                                                  ["curr_x"], )
Combined_Plate_reduced_AlSt_TrainVal = DataclassCombinedTrainVal('Plate_reduced_AlSt_TrainVal', folder_data,
                                                  ['AL_2007_T4_Plate_SF', 'AL_2007_T4_Plate_Depth', 'S235JR_Plate_SF', 'S235JR_Plate_Depth'
                                                        ], # 'Laufrad_Durchlauf_1', 'Laufrad_Durchlauf_2'
                                                  dataPaths_Test,
                                                  ["curr_x"], )
Combined_PlateGear_reduced_TrainVal = DataclassCombinedTrainVal('PlateGear_reduced_TrainVal', folder_data,
                                                  ['AL_2007_T4_Plate_SF', 'AL_2007_T4_Plate_Depth', 'AL_2007_T4_Gear_SF', 'AL_2007_T4_Gear_Depth'
                                                        ], # 'Laufrad_Durchlauf_1', 'Laufrad_Durchlauf_2'
                                                  dataPaths_Test,
                                                  ["curr_x"], )
Combined_Plate_TrainVal_CONTDEV = DataclassCombinedTrainVal('Plate_TrainVal_CONTDEV', '..\\..\\DataSets\DataMatched',
                                                    ['AL_2007_T4_Plate_Depth', 'AL_2007_T4_Plate_SF'],
                                                    [ 'AL_2007_T4_Gear_Normal_3.csv','AL_2007_T4_Plate_Normal_3.csv', 'S235JR_Gear_Normal_3.csv','S235JR_Plate_Normal_3.csv'],
                                                    ["curr_x"], header = ["v_sp", "v_x", "v_y", "v_z",
                                                                          "a_x", "a_y", "a_z", "a_sp",
                                                                          "f_x_sim", "f_y_sim", "f_z_sim", "f_sp_sim",
                                                                          "materialremoved_sim",
                                                                          "CONT_DEV_X", "CONT_DEV_Y", "CONT_DEV_Z"])
Combined_OldData = DataClass('OldData', '..\\..\\DataSets\OldDataSets',
                            [   'CMX_Alu_Tr_Air_2_alldata_allcurrent.pkl',
                                                'CMX_Alu_Tr_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                                'I40_Alu_Tr_Air_2_alldata_allcurrent.pkl',
                                                'I40_Alu_Tr_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                'CMX_St_Tr_Air_2_alldata_allcurrent.pkl',
                                'CMX_St_Tr_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                'I40_St_Tr_Air_2_alldata_allcurrent.pkl',
                                'I40_St_Tr_Mat_2_alldata_allforces_MRR_allcurrent.pkl'
                                ],
                             [ 'CMX_Alu_Val_Air_2_alldata_allcurrent.pkl',
                                                'CMX_Alu_Val_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                                'I40_Alu_Val_Air_2_alldata_allcurrent.pkl',
                                                'I40_Alu_Val_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                                'CMX_Alu_St_Air_2_alldata_allcurrent.pkl',
                                                'CMX_Alu_St_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                                'I40_Alu_St_Air_2_alldata_allcurrent.pkl',
                                                'I40_Alu_St_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                               ],
                             ['CMX_Alu_Tr_Air_3_alldata_allcurrent.pkl',
                                             'CMX_Alu_Tr_Mat_3_alldata_allforces_MRR_allcurrent.pkl',
                                             'I40_Alu_Tr_Air_3_alldata_allcurrent.pkl',
                                             'I40_Alu_Tr_Mat_3_alldata_allforces_MRR_allcurrent.pkl',
                                             'CMX_Alu_St_Air_3_alldata_allcurrent.pkl',
                                             'CMX_Alu_St_Mat_3_alldata_allforces_MRR_allcurrent.pkl',
                                             'I40_Alu_St_Air_3_alldata_allcurrent.pkl',
                                             'I40_Alu_St_Mat_3_alldata_allforces_MRR_allcurrent.pkl'
                                              ]
                             )
Combined_OldData_noAir = DataClass('OldData no Air', '..\\..\\DataSets\OldDataSets',
                            [   'CMX_Alu_Tr_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                                'I40_Alu_Tr_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                                'CMX_St_Tr_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                                'I40_St_Tr_Mat_2_alldata_allforces_MRR_allcurrent.pkl'
                                            ],
                             [ 'CMX_Alu_Val_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                                'I40_Alu_Val_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                                'CMX_Alu_St_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                                'I40_Alu_St_Mat_2_alldata_allforces_MRR_allcurrent.pkl',
                                                ],
                             ['CMX_Alu_Tr_Mat_3_alldata_allforces_MRR_allcurrent.pkl',
                                             'I40_Alu_Tr_Mat_3_alldata_allforces_MRR_allcurrent.pkl',
                                             'CMX_Alu_St_Mat_3_alldata_allforces_MRR_allcurrent.pkl',
                                             'I40_Alu_St_Mat_3_alldata_allforces_MRR_allcurrent.pkl'
                                              ]
                             )

Combined_KL = DataClass('KL', folder_data,
                        dataPaths_Val_KL,
                        dataPaths_Train_Plate,
                        dataPaths_Test,
                        ["curr_x"], )

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
                data[f'v_{axis}'] = data[pos].diff()
                data[f'a_{axis}'] = data[f'v_{axis}'].diff()
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

def load_filtered_data(data_params: BaseDataClass, past_values=2, future_values=2, window_size=1):
    """
    Loads and preprocesses data for training, validation, and testing.
    Does not create ml vector.
    No separation between x and y.

    Parameters
    ----------
    data_params : BaseDataClass
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
    training_datas = apply_action(fulltrainingdatas, lambda data: data[HEADER] )
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

def get_test_data_as_pd(data_params: BaseDataClass, past_values=2, future_values=2, window_size=1):
    """
    Loads and preprocesses test data for evaluation purposes.
    Applies a rolling mean to the data and adjusts the dataset based on specified past and future values.

    Parameters
    ----------
    data_params : BaseDataClass
        A DataClass containing the data parameters for loading the dataset.
    past_values : int, optional
        The number of past values to consider for each sample. The default is 2.
    future_values : int, optional
        The number of future values to predict for each sample. The default is 2.
    window_size : int, optional
        The size of the sliding window for calculating the rolling mean. The default is 1.
        If 0 or negative, it is adjusted to a positive value.

    Returns
    -------
    tuple
        A tuple containing the preprocessed test data and base names of the test files:
        - test_datas : list of pd.DataFrame
            Preprocessed test data with rolling mean applied.
        - base_names : list of str
            Base names of the test files without the index part.
    """
    if window_size == 0:
        window_size = 1
    elif window_size < 0:
        window_size = abs(window_size)

    # Getting validation data to right format:
    fulltestdatas = read_fulldata(data_params.testing_data_paths, data_params.folder)

    # Apply rolling mean to each DataFrame in the lists
    test_datas = apply_action(fulltestdatas, lambda data: data[HEADER].rolling(window=window_size, min_periods=1).mean())

    if past_values + future_values != 0:
        test_datas = apply_action(test_datas, lambda target: target.iloc[past_values:-future_values])

    base_names = []
    for path in data_params.testing_data_paths:
        file_name = os.path.basename(path)
        name_without_extension = os.path.splitext(file_name)[0]
        # Split the name by underscore and remove the last part (the index)
        base_names.append('_'.join(name_without_extension.split('_')[:-1]))
    return test_datas, base_names

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
            data_classes.append(DataclassCombinedTrainVal(base_name, folder_path, training_name, validation_name, test_name))

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
        data_classes.append(DataclassCombinedTrainVal(base_name, folder_path, training_name, validation_name, test_name))

    return data_classes

def create_DataClasses_from_base_names(folder_path, training_base_names, validation_base_names, test_base_names, name):
    file_names = get_csv_files(folder_path)

    training_name = [file_name for file_name in file_names if any(name in file_name for name in training_base_names)]
    validation_name = [file_name for file_name in file_names if any(name in file_name for name in validation_base_names)]
    test_name = [file_name for file_name in file_names if any(name in file_name for name in test_base_names)]

    return DataclassCombinedTrainVal(name, folder_path, training_name, validation_name, test_name)

def save_data(data_list, file_paths):
    """
    Saves the data to a CSV file. If the file already exists, existing columns
    are overwritten and new columns are added.

    Parameters:
        data_list (list of pd.DataFrame): The new data to be saved.
        file_paths (list of string): Strings containing the file path.

    Returns:
        None
    """
    if type(data_list) == list or type(file_paths) == list:
        assert len(data_list) == len(file_paths)

    for data, file_path in zip(data_list, file_paths):
        # Check if the file already exists
        if os.path.exists(file_path):
            # Read the existing file
            existing_data = pd.read_csv(file_path)

            # Update existing columns and add new columns
            for col in data.columns:
                existing_data[col] = data[col]

            # Save the updated data back to the file
            existing_data.to_csv(file_path, index=False, header=True)
        else:
            # If the file does not exist, create a new file
            data.to_csv(file_path, index=False, header=True)

def add_pd_to_csv(data_list, file_paths, headers):
    """
    Saves the data to a CSV file. If the file already exists, existing columns
    are overwritten and new columns are added.

    Parameters:
        data_list (list of pd.DataFrame): The new data to be saved.
        file_paths (list of string): Strings containing the file path.
        headers (list of list of string): Headers for each data.

    Returns:
        None
    """
    assert len(data_list) == len(file_paths)
    assert len(data_list) == len(headers)

    for data, file_path, header in zip(data_list, file_paths, headers):
        # Convert input data and header to DataFrame
        data_df = pd.DataFrame(data, columns=header)

        if os.path.exists(file_path):
            # Read the existing file
            existing_data = pd.read_csv(file_path)

            # Check if header columns already exist
            if all(col in existing_data.columns for col in header):
                # Update the existing columns with new data
                existing_data.update(data_df)
            else:
                # Concatenate new columns to the existing data
                existing_data = pd.concat([existing_data, data_df], axis=1)

            # Save the combined data
            existing_data.to_csv(file_path, index=False)
            print(f"{file_path} updated")
        else:
            # Create the file with the new data
            data_df.to_csv(file_path, index=False)
            print(f"{file_path} created")

def preprocessing(self, X, y, n=12):
    """
    Remove outliers from the input data and adjust the corresponding y values.

    Parameters:
    X (DataFrame or Series): The input data.
    y (Series or DataFrame): The corresponding target values.
    n (int): The number of standard deviations to consider for outlier removal. Default is 12.

    Returns:
    Tuple: The cleaned input data and the adjusted target values.
    """
    if isinstance(y, pd.Series):
        # Calculate the mean and standard deviation for the target values
        mean = y.mean()
        std = y.std()

        # Identify outliers in the target values
        outliers = np.abs(y - mean) > n * std
    elif isinstance(y, pd.DataFrame):
        # Calculate the mean and standard deviation for each feature
        mean = y.mean(axis=0)
        std = y.std(axis=0)

        # Identify outliers
        outliers = np.abs(y - mean) > n * std
        outliers = outliers.any(axis=1)
    else:
        raise ValueError("y must be a pandas Series or DataFrame")

    # Remove rows that contain outliers
    x_cleaned = X[~outliers]
    y_cleaned = y[~outliers]

    return x_cleaned, y_cleaned

def load_data(data_params: DataclassCombinedTrainVal, past_values=2, future_values=2, window_size=1, keep_separate=False, N=3, do_preprocessing=True, n=12):
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
    fulltestdatas = read_fulldata(data_params.testing_data_paths, data_params.folder)
    test_datas =  apply_action(fulltestdatas, lambda data: data[HEADER_x].rolling(window=window_size, min_periods=1).mean())
    X_test =  apply_action(test_datas, lambda data: create_full_ml_vector_optimized(past_values, future_values, data))
    test_targets =  apply_action(fulltestdatas, lambda data: data[data_params.target_channels])
    y_test =  apply_action(test_targets, lambda target: target.rolling(window=window_size, min_periods=1).mean())
    if past_values + future_values != 0:
        y_test =  apply_action(y_test, lambda target: target.iloc[past_values:-future_values])

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

        file_datas = read_fulldata(files, data_params.folder)
        file_datas_x = apply_action(file_datas, lambda data: data[HEADER_x].rolling(window=window_size,
                                                                                                min_periods=1).mean())
        file_datas_y = apply_action(file_datas,
                                          lambda data: data[data_params.target_channels].rolling(window=window_size,
                                                                                                 min_periods=1).mean())

        X_files = apply_action(file_datas_x,
                                     lambda data: create_full_ml_vector_optimized(past_values, future_values,
                                                                                        data))
        y_files = apply_action(file_datas_y, lambda target: target.iloc[
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

            if do_preprocessing:
                X_train_parts, y_train_parts = zip(*[preprocessing(X, y, n) for X, y in zip(X_train_parts, y_train_parts)])
                X_val_parts, y_val_parts = zip(*[preprocessing(X, y, n) for X, y in zip(X_val_parts, y_val_parts)])

            all_X_train.extend(X_train_parts)
            all_y_train.extend(y_train_parts)
            all_X_val.extend(X_val_parts)
            all_y_val.extend(y_val_parts)

        toggle = not toggle  # Umschalten für nächste Datei

    if len(X_test) <= 1:
        X_test = pd.concat(X_test).reset_index(drop=True)
        y_test = pd.concat(y_test).reset_index(drop=True)

    if keep_separate:
        return all_X_train, all_X_val, X_test, all_y_train, all_y_val, y_test
    else:
        X_train = pd.concat(all_X_train).reset_index(drop=True)
        y_train = pd.concat(all_y_train).reset_index(drop=True)
        X_val = pd.concat(all_X_val).reset_index(drop=True)
        y_val = pd.concat(all_y_val).reset_index(drop=True)

        return X_train, X_val, X_test, y_train, y_val, y_test

# Berechne MSE und Standardabweichung pro Modell und Methode
def calculate_mae_and_std(predictions_list, true_values, n_drop_values=10, center_data = False):
    mae_values = []

    for pred in predictions_list:
        # Werte kürzen
        pred_trimmed = pred[:-n_drop_values]
        true_trimmed = true_values[:-n_drop_values]

        if center_data:
            # Zentrierung
            mean = np.mean(true_trimmed)
            pred_centered = pred_trimmed - mean
            true_centered = true_trimmed - mean
        else:
            pred_centered = pred_trimmed
            true_centered = true_trimmed

        mae = np.mean(np.abs(pred_centered.squeeze() - true_centered.squeeze()))
        mae_values.append(mae)
    pred_mean = np.mean(predictions_list, axis=0)
    mae_ensemble = np.mean(np.abs(pred_mean.squeeze() - true_values.squeeze()))

    return np.mean(mae_values), np.std(mae_values), mae_ensemble

def load_data_with_material_check(data_params: DataclassCombinedTrainVal, past_values=2, future_values=2, window_size=1, keep_separate=False, N=3, do_preprocessing=True, n=12):
    """
    Loads and preprocesses data for training, validation, and testing, and checks for material parameters.

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
        X_train, X_val, X_test, y_train, y_val, y_test with respective hardness values
    """

    material_parameters = {
        "AL_2007_T4": {
            "hardness": 320,
            "tensile_strength": 370,
            "yield_strength": 250,
            "thermal_conductivity": 145,
            "modulus_of_elasticity": 72.5,
            "density": 2.85
        },
        "S235JR": {
            "hardness": 120,
            "tensile_strength": 570,
            "yield_strength": 300,
            "thermal_conductivity": 54,
            "modulus_of_elasticity": 212,
            "density": 7.85
        }
    }

    def get_material_hardness(file_path):
        for material in material_parameters.keys():
            if material in file_path:
                return material_parameters[material]["hardness"]
        raise ValueError(f"No valid material found in the file path: {file_path}")

    # Load test data
    fulltestdatas = read_fulldata(data_params.testing_data_paths, data_params.folder)
    test_datas = apply_action(fulltestdatas, lambda data: data[HEADER_x].rolling(window=window_size, min_periods=1).mean())
    X_test = apply_action(test_datas, lambda data: create_full_ml_vector_optimized(past_values, future_values, data))
    test_targets = apply_action(fulltestdatas, lambda data: data[data_params.target_channels])
    y_test = apply_action(test_targets, lambda target: target.rolling(window=window_size, min_periods=1).mean())
    if past_values + future_values != 0:
        y_test = apply_action(y_test, lambda target: target.iloc[past_values:-future_values])

    # Get hardness for each test file
    test_hardness = [get_material_hardness(path) for path in data_params.testing_data_paths]

    # Namen der Testdateien zum Ausschluss
    test_files = set(os.path.basename(p) for p in data_params.testing_data_paths)

    # Trainings- und Validierungsdaten vorbereiten
    all_X_train, all_y_train, all_hardness_train = [], [], []
    all_X_val, all_y_val, all_hardness_val = [], [], []

    toggle = True  # Start mit gerade = Train

    for name in data_params.training_validation_datas:
        pattern = f"{name}_*.csv"
        files = glob.glob(os.path.join(data_params.folder, pattern))
        files = [f for f in files if os.path.basename(f) not in test_files]

        file_datas = read_fulldata(files, data_params.folder)
        file_datas_x = apply_action(file_datas, lambda data: data[HEADER_x].rolling(window=window_size,
                                                                                            min_periods=1).mean())
        file_datas_y = apply_action(file_datas,
                                          lambda data: data[data_params.target_channels].rolling(window=window_size,
                                                                                             min_periods=1).mean())

        X_files = apply_action(file_datas_x,
                                     lambda data: create_full_ml_vector_optimized(past_values, future_values,
                                                                                        data))
        y_files = apply_action(file_datas_y, lambda target: target.iloc[
                                                                  past_values:-future_values] if past_values + future_values != 0 else target)

        for file_path, X_df, y_df in zip(files, X_files, y_files):
            hardness = get_material_hardness(file_path)
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

            if do_preprocessing:
                X_train_parts, y_train_parts = zip(*[preprocessing(X, y, n) for X, y in zip(X_train_parts, y_train_parts)])
                X_val_parts, y_val_parts = zip(*[preprocessing(X, y, n) for X, y in zip(X_val_parts, y_val_parts)])

            all_X_train.extend(X_train_parts)
            all_y_train.extend(y_train_parts)
            all_hardness_train.extend([hardness] * len(X_train_parts))
            all_X_val.extend(X_val_parts)
            all_y_val.extend(y_val_parts)
            all_hardness_val.extend([hardness] * len(X_val_parts))

        toggle = not toggle  # Umschalten für nächste Datei

    if len(X_test) <= 1:
        X_test = pd.concat(X_test).reset_index(drop=True)
        y_test = pd.concat(y_test).reset_index(drop=True)

    if keep_separate:
        return (all_X_train, all_hardness_train), (all_X_val, all_hardness_val), (X_test, test_hardness), all_y_train, all_y_val, y_test
    else:
        X_train = pd.concat(all_X_train).reset_index(drop=True)
        y_train = pd.concat(all_y_train).reset_index(drop=True)
        counter = Counter(all_hardness_train)
        hardness_train = counter.most_common(1)[0][0]

        X_val = pd.concat(all_X_val).reset_index(drop=True)
        y_val = pd.concat(all_y_val).reset_index(drop=True)
        counter = Counter(all_hardness_val)
        hardness_val = counter.most_common(1)[0][0]

        return (X_train, hardness_train), (X_val, hardness_val), (X_test, test_hardness), y_train, y_val, y_test