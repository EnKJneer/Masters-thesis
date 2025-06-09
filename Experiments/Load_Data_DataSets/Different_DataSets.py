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

class CustomDataLoader(hdata.DataclassCombinedTrainVal):
    def load_data(self):
        # Load test data
        fulltestdatas = hdata.read_fulldata(self.testing_data_paths, self.folder)
        test_datas = hdata.apply_action(fulltestdatas, lambda data: data[hdata.HEADER_x].rolling(window=self.window_size, min_periods=1).mean())
        X_test = hdata.apply_action(test_datas, lambda data: hdata.create_full_ml_vector_optimized(self.past_values, self.future_values, data))
        test_targets = hdata.apply_action(fulltestdatas, lambda data: data[self.target_channels])
        y_test = hdata.apply_action(test_targets, lambda target: target.rolling(window=self.window_size, min_periods=1).mean())

        if self.past_values + self.future_values != 0:
            y_test = hdata.apply_action(y_test, lambda target: target.iloc[self.past_values:-self.future_values])

        threshold = 0.01  # Define a threshold for 'v_x' being approximately 0

        # Initialisiere leere Listen für die gefilterten Daten
        filtered_y_test = []
        filtered_X_test = []

        # Iteriere über die DataFrames und Serien und filtere sie
        for df, series in zip(X_test, y_test):
            # Stelle sicher, dass die Indizes übereinstimmen
            df = df.reset_index(drop=True)
            series = series.reset_index(drop=True)

            # Erstelle eine Maske basierend auf der Bedingung
            mask = abs(df['v_x_1_current']) < threshold

            # Filtere den DataFrame basierend auf der Maske
            filtered_df = df[mask].reset_index(drop=True)

            # Filtere die Serie basierend auf der Maske
            filtered_series = series[mask].reset_index(drop=True)

            filtered_y_test.append(filtered_series)
            filtered_X_test.append(filtered_df)

        # Überschreibe die ursprünglichen Listen mit den gefilterten Daten
        y_test = filtered_y_test
        X_test = filtered_X_test

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

            file_datas = hdata.read_fulldata(files, self.folder)
            file_datas_x = hdata.apply_action(file_datas, lambda data: data[hdata.HEADER_x].rolling(window=self.window_size, min_periods=1).mean())
            file_datas_y = hdata.apply_action(file_datas, lambda data: data[self.target_channels].rolling(window=self.window_size, min_periods=1).mean())

            X_files = hdata.apply_action(file_datas_x, lambda data: hdata.create_full_ml_vector_optimized(self.past_values, self.future_values, data))
            y_files = hdata.apply_action(file_datas_y, lambda target: target.iloc[self.past_values:-self.future_values] if self.past_values + self.future_values != 0 else target)

            # Filtere die Daten basierend auf der Bedingung
            filtered_y_files = []
            filtered_X_files = []

            for df, series in zip(X_files, y_files):
                df = df.reset_index(drop=True)
                series = series.reset_index(drop=True)

                mask = abs(df['v_x_1_current']) < threshold
                filtered_df = df[mask].reset_index(drop=True)
                filtered_series = series[mask].reset_index(drop=True)
                filtered_y_files.append(filtered_series)
                filtered_X_files.append(filtered_df)

            for X_df, y_df in zip(filtered_X_files, filtered_y_files):
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


if __name__ == "__main__":
    """ Constants """
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 2

    window_size = 10
    past_values = 2
    future_values = 2

    model_rf = mrf.get_reference()  #mnn.Net(None, 1, 200, 2, learning_rate=0.001, name='Neural_Net_large') #

    #Combined_Gear,Combined_KL
    dataClass_1 = hdata.Combined_Plate_TrainVal
    dataClass_1.window_size = window_size
    dataClass_1.past_values = past_values
    dataClass_1.future_values = future_values

    folder_data = '..\\..\\DataSets\DataFiltered'
    dataPaths_Test = ['AL_2007_T4_Gear_Normal_3.csv', 'AL_2007_T4_Plate_Normal_3.csv', 'S235JR_Gear_Normal_3.csv',
                      'S235JR_Plate_Normal_3.csv']
    Combined_PK_TrainVal = CustomDataLoader('PK_TrainVal', folder_data,
                                                     ['AL_2007_T4_Plate', 'AL_2007_T4_Plate_Depth',
                                                      'AL_2007_T4_Plate_SF',
                                                      'Kühlgrill_Mat_S2800', 'Kühlgrill_Mat_S3800',
                                                      'Kühlgrill_Mat_S4700',
                                                      ],  # 'Laufrad_Durchlauf_1', 'Laufrad_Durchlauf_2'
                                                     dataPaths_Test,
                                                     ["curr_x"], )

    dataClass_2 = Combined_PK_TrainVal
    dataClass_2.window_size = window_size
    dataClass_2.past_values = past_values
    dataClass_2.future_values = future_values
    dataSets_list = [dataClass_2] # dataClass_1

    experiment_results = hexp.run_experiment(dataSets_list, True, False, [model_rf],
                        NUMBEROFEPOCHS, NUMBEROFMODELS, past_values, future_values,n_drop_values=25,
                        plot_types=['heatmap', 'prediction_overview'])
    # 'heatmap_std', , 'geometry_mae', 'force_mae', 'mrr_mae'

    # Zugriff auf Ergebnisse
    print(f"Experiment gespeichert in: {experiment_results['results_dir']}")
    print(f"Anzahl Plots erstellt: {sum(len(paths) for paths in experiment_results['plot_paths'].values())}")