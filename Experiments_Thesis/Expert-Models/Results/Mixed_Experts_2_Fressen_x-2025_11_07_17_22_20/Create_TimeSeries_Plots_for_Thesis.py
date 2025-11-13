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

import Helper.handling_hyperopt as hyperopt
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
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
            'ycolname': 'ST_Plate_Notch_Mixed_Experts_2',
            'ylabel': 'Experten Modell'
        }
    ]

    cut_offs = {'S235JR_Plate': (2, 40), # ST Plate
               'S235JR_Gear': (2, 24),  # ST Gear
               'AL2007T4_Plate': (2, 67),  # AL Plate
               'AL2007T4_Gear': (2, 40),  # AL Gear
                }

    for material in materials:

        for geometry in geometries:

            data = []
            for path in paths:
                cut_off = cut_offs[f'{material}_{geometry}']
                if f'{material}_{geometry}' == 'AL2007T4_Gear':
                    file = f'DMC60H_{material}_{geometry}_Depth_3.csv'
                else:
                    file = f'DMC60H_{material}_{geometry}_Normal_3.csv'
                df = pd.read_csv(f'{path}/{file}')
                df = df.iloc[cut_off[0]*50:cut_off[1]*50]
                data.append(df)

            df = pd.concat(data, axis=1).reset_index(drop=True)

            # Doppelte Spalten (identisch) entfernen
            df = df.loc[:, ~df.columns.duplicated()]
            if material == 'AL2007T4':
                mat = 'Aluminium'
            else:
                mat = 'Stahl'

            hplot.plot_time_series(df, f'{mat} {geometry}:\nStromverlauf der Vorschubachse in x-Richtung',
                             f'Verlauf_{material}_{geometry}',
                             col_name='curr_x', dpi=600,
                             y_configs=y_configs, path='Plots_Thesis')


