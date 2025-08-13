import copy
import datetime
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_experiment as hexp
import Models.model_random_forest as mrf
import Models.model_neural_net as mnn
import Models.model_physical as mphy

# Zur Archivierung: Verwendete Funktion
def load_data_from_path(self, data_paths):
    fulldatas = hdata.read_fulldata(data_paths, self.folder)
    datas = hdata.apply_action(fulldatas,
                         lambda data: data[self.header].rolling(window=self.window_size, min_periods=1).mean())

    if self.columns_to_integrate:
        datas = hdata.apply_action(datas, lambda data: self.integrate_columns(data))

    X = hdata.apply_action(datas,
                     lambda data: hdata.create_full_ml_vector_optimized(self.past_values, self.future_values, data))
    targets = hdata.apply_action(fulldatas, lambda data: data[self.target_channels])
    Y = hdata.apply_action(targets, lambda target: target.rolling(window=self.window_size, min_periods=1).mean())

    if self.use_filter:
        X = hdata.apply_action(X, lambda data: self.apply_lowpass_filter(data, self.cutoff, self.filter_order))
        Y = hdata.apply_action(Y, lambda data: self.apply_lowpass_filter(data, self.cutoff, self.filter_order))

    if self.add_sign_hold:
        X = hdata.apply_action(X, lambda data: self.add_z_to_data(data))

    if self.add_sign_y:
        # Add the sign of the previous y value as a new column in x
        for x_df, y_df in zip(X, Y):
            for col in y_df.columns:
                y_col_sign = f"{col}_sign"
                x_df[y_col_sign] = y_df[col].shift(1).apply(lambda x: 0 if pd.isna(x) else x).fillna(0)

    if self.past_values + self.future_values != 0:
        Y = hdata.apply_action(Y, lambda target: target.iloc[self.past_values:-self.future_values])

    if len(data_paths) <= 1:
        X = pd.concat(X).reset_index(drop=True)
        Y = pd.concat(Y).reset_index(drop=True)

    return X, Y

if __name__ == '__main__':

    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataclass1 = copy.copy(dataSet)
    dataclass1.name = 'ohne y-1'
    dataclass2 = copy.copy(dataSet)
    dataclass2.name = 'mit y-1'
    dataclass2.add_sign_y = True
    # dataclass2 = hdata.Combined_Plate_TrainVal
    dataClasses = [dataclass1, dataclass2]
    for dataclass in dataClasses:
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values
        dataclass.add_padding = True

    model_rf = mrf.RandomForestModel(n_estimators= 52,max_features= 500, min_samples_split= 67,
                    min_samples_leaf= 4)
    model_rnn = mnn.RNN(learning_rate= 0.04834201195017264, n_hidden_size= 94, n_hidden_layers= 1,
                    activation= 'Sigmoid', optimizer_type= 'quasi_newton')
    models = [model_rnn, model_rf]

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name='y_previous')
