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

    paths = ['Results/Physical_Model_Erd-2025_09_27_17_34_33/Predictions',
             'Results/Physical_Model_Thermo-2025_10_21_11_28_35/Predictions']

    y_configs = [
        {
            'ycolname': 'ST_Plate_Notch_Erd',
            'ylabel': 'Modell-Erd'
        },
        {
            'ycolname': 'ST_Plate_Notch_ThermodynamicModel',
            'ylabel': 'Verbessertes Modell'
        },
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
            hplottime.plot_time_series(df, f'{mat} {geometry}:\nStromverlauf der Vorschubachse in x-Richtung',
                             f'Verlauf_{material}_{geometry}',
                             col_name='curr_x', label='Strom-Messwert', dpi=600,
                                       y_configs=y_configs,
                                       fontsize_axis=22, fontsize_axis_label=24, fontsize_label=26,
                                       fontsize_title=28,
                                       )


