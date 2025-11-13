import copy
import json
import os
import ast
import re

import shap
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
import seaborn as sns
from numpy.exceptions import AxisError
from numpy.f2py.auxfuncs import throw_error
from sklearn.metrics import mean_absolute_error

import Helper.handling_hyperopt as hyperopt
import Helper.handling_data as hdata
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
from matplotlib.colors import LinearSegmentedColormap
import Helper.handling_timeseries_plots as hplottime

if __name__ == '__main__':

    # Beispielaufruf der Funktion
    materials = ['S235JR', 'AL2007T4']
    geometries = ['Plate', 'Gear']
    axis = 'x'

    paths = ['Referenzmodelle/Results/Recurrent_Neural_Net-2025_10_02_23_09_20/Predictions',
             'Referenzmodelle/Results/Random_Forest-2025_10_20_20_08_14/Predictions',
             'ML-Modelle/Results/Recurrent_Neural_Net-2025_10_02_17_09_17/Predictions',
             'ML-Modelle/Results/Recurrent_Neural_Net-2025_10_02_16_20_24/Predictions',
             'ML-Modelle/Results/Random_Forest-2025_10_20_21_18_53/Predictions'
             ]

    models = [['Reference_Random_Forest_TPESampler', 'ST_Plate_Notch_Random_Forest_GridSampler', 'Random_Forest'],
              ['Reference_Recurrent_Neural_Net_TPESampler', 'ST_Plate_Notch_Recurrent_Neural_Net_GridSampler',
               'Recurrent_Neural_Net']]

    y_configs_rf = [
        {
            'ycolname': 'Reference_Random_Forest_TPESampler',
            'ylabel': 'Referenzdaten'
        },
        {
            'ycolname': 'ST_Plate_Notch_Random_Forest_GridSampler',
            'ylabel': 'Optimierte Daten'
        },
    ]

    y_configs_rnn = [
        {
            'ycolname': 'Reference_Recurrent_Neural_Net_TPESampler',
            'ylabel': 'Referenzdaten'
        },
        {
            'ycolname': 'ST_Plate_Notch_Recurrent_Neural_Net_GridSampler',
            'ylabel': 'Optimierte Daten'
        },
    ]
    y_configs = [y_configs_rf, y_configs_rnn]
    names = ['Random_Forest', 'Recurrent_Neural_Net']

    for material in materials:

        for geometry in geometries:

            data = []
            for path in paths:

                file = f'DMC60H_{material}_{geometry}_Normal_3.csv'

                data.append(pd.read_csv(f'{path}/{file}'))

            df = pd.concat(data, axis=1).reset_index(drop=True)

            # Doppelte Spalten (identisch) entfernen
            df = df.loc[:, ~df.columns.duplicated()]
            if material == 'AL2007T4':
                mat = 'Aluminium'
            else:
                mat = 'Stahl'



            for model, y_config in zip(names, y_configs):
                name = model.replace('_', ' ')
                name = name.replace('Recurrent Neural Net', 'Rekurrentes neuronales Netz')
                hplottime.plot_time_series(df, f'{mat} {geometry} {name}:\nStromverlauf der Vorschubachse in x-Richtung',
                                 f'{model}_Verlauf_{material}_{geometry}',
                                 col_name='curr_x', label='Strom-Messwerte', dpi=600,
                                 y_configs=y_config,
                                           fontsize_axis=22, fontsize_axis_label=24, fontsize_label=26,
                                           fontsize_title=28,
                                           )
