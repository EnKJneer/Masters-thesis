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
    # Liste für alle DataFrames mit MAE/StdDev
    all_mae_std_dfs = []

    model_prefixes_1 = [
        'ST_Plate_Notch_Recurrent_Neural_Net_RandomSampler',
        'ST_Plate_Notch_Recurrent_Neural_Net_GridSampler',

    ]
    model_prefixes_2 = ['ST_Plate_Notch_Recurrent_Neural_Net_TPESampler']

    model_prefixes_list = [model_prefixes_1, model_prefixes_2]

    paths = ['Recurrent_Neural_Net-2025_10_02_16_20_24',
             'Recurrent_Neural_Net-2025_10_02_17_09_17']

    for path, model_prefixes in zip(paths, model_prefixes_list):
        paths_target = path+'/Predictions'
        for file in os.listdir(paths_target):
            if file.endswith('.csv'):
                df = pd.read_csv(f"{paths_target}/{file}")
                # MAE und StdDev für die Sampler berechnen
                mae_std_df = hplot.calculate_mae_and_std(df, file, model_prefixes=model_prefixes)
                all_mae_std_dfs.append(mae_std_df)

    # Alle DataFrames zu einem einzigen DataFrame kombinieren
    combined_mae_std_df = pd.concat(all_mae_std_dfs, ignore_index=True)

    # HeatmapPlotter initialisieren
    plotter = hplot.HeatmapPlotter(output_dir='Plots_Thesis')

    # Heatmap erstellen
    plot_paths = plotter.create_plots(
        df=combined_mae_std_df,
        title='Rekurrentes neuronales Netz',
        filename='RNN',
    )

    print(f"Heatmaps wurden erstellt: {plot_paths}")

    # ModelHeatmapPlotter initialisieren
    plotter = hplot.ModelHeatmapPlotter(output_dir='Plots_Thesis')

    # Heatmaps für jedes Modell erstellen
    plot_paths = plotter.create_plots(df=combined_mae_std_df)

    print(f"Heatmaps wurden erstellt: {plot_paths}")
