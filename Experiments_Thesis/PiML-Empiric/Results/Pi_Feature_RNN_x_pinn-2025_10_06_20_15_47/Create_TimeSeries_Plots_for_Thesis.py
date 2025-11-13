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
from matplotlib.patches import Patch
from numpy.exceptions import AxisError
from numpy.f2py.auxfuncs import throw_error
from sklearn.metrics import mean_absolute_error

import Helper.handling_timeseries_plots as hplottime
import Helper.handling_hyperopt as hyperopt
import Helper.handling_data as hdata
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
from matplotlib.colors import LinearSegmentedColormap

if __name__ == '__main__':

    # Beispielaufruf der Funktion
    materials = ['S235JR', 'AL2007T4']
    geometries = ['Plate', 'Gear']

    paths = ['Predictions']

    y_configs = [
        {
            'ycolname': 'Normal_Recurrent_Neural_Net',
            'ylabel': 'RNN'
        },
        {
            'ycolname': 'Sign_Hold_x_Recurrent_Neural_Net',
            'ylabel': 'RNN mit reduzierten Featuren'
        }
    ]

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

            hplottime.plot_time_series_sections(df, f'{mat} {geometry}:\nStromverlauf der Vorschubachse in x-Richtung',
                             f'Verlauf_{material}_{geometry}',
                             col_name='curr_x', dpi=600,
                             y_configs=y_configs, path='Plots_Thesis',
                                       fontsize_axis=22, fontsize_axis_label=24,
                                       fontsize_title=28,
                                       )


