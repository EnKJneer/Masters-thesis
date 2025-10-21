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
    # Beispielaufruf der Funktion
    materials = ['S235JR', 'AL2007T4']
    geometries = ['Plate', 'Gear']
    version = 3

    for material in materials:
        for geometry in geometries:
            df = pd.read_csv(f'Data/DMC60H_{material}_{geometry}_Blowhole_{version}.csv')

            if material == 'AL2007T4':
                mat = 'Aluminium'
            else:
                mat = 'Stahl'
            hplot.plot_time_series(df, f'{mat} {geometry}: Stromverlauf der Vorschubachse in x-Richtung',
                             f'{material}_{geometry}_Anomalie',
                             col_name='curr_x', label='$I$\nin A', dpi=600)