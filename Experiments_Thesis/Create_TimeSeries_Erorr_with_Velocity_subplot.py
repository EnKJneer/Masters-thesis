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
import Helper.handling_plots as hplot
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
from matplotlib.colors import LinearSegmentedColormap

if __name__ == '__main__':
    # Beispielaufruf mit der neuen y_configs-Struktur
    materials = ['S235JR', 'AL2007T4']
    geometries = ['Plate', 'Gear']
    paths = [
        #'ML-Modelle/Results/Recurrent_Neural_Net-2025_10_02_16_20_24/Predictions',
        'ML-Modelle/Results/Random_Forest-2025_10_20_21_18_53/Predictions',
        #'PiML-Erd/Results/ThermodynamicModel_Residual-2025_09_29_21_42_39/Predictions'
    ]

    for material in materials:
        for geometry in geometries:
            data = []
            for idx, path in enumerate(paths):
                file = f'DMC60H_{material}_{geometry}_Normal_3.csv'
                df = pd.read_csv(os.path.join(path, file))
                data.append(pd.read_csv(f'{path}/{file}'))

            df = pd.concat(data, axis=1).reset_index(drop=True)
            df = df.loc[:, ~df.columns.duplicated()]

            mat = 'Aluminium' if material == 'AL2007T4' else 'Stahl'

            # Definition der y-Konfigurationen als Liste von Dictionaries
            y_configs = [
                {
                    'ycolname': 'ST_Plate_Notch_Random_Forest_GridSampler',
                    'ylabel': 'Random Forest'
                },
                {
                    'ycolname': 'ST_Plate_Notch_Recurrent_Neural_Net_GridSampler',
                    'ylabel': 'RNN'
                },
                {
                    'ycolname': 'ST_Plate_Notch_Hybrid Erd Random Forest',
                    'ylabel': 'Residual-RF'
                }
            ]

            hplot.plot_time_series_error_with_sections(
                df,
                f'{mat} {geometry}: Stromverlauf und Vorschubgeschwindigkeit',
                f'Fehler_{material}_{geometry}_mit_Vorschub',
                col_name='curr_x',
                label='$I$\nin A',
                v_colname='v_x',
                v_label='Vorschubgeschwindigkeit',
                v_axis='$v$\nin m/s',
                y_configs=y_configs,
                dpi=600
            )
