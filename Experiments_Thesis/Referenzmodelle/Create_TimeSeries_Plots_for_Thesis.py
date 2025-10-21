import copy
import json
import os

import pandas as pd

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

    paths = ['Results/Random_Forest-2025_10_20_20_08_14/Predictions'
             ] #'Results/Recurrent_Neural_Net-2025_10_02_23_09_20/Predictions',

    y_configs = [
        {
            'ycolname': 'Reference_Random_Forest',
            'ylabel': 'Random Forest (mean +/- std)'
        },
        {
            'ycolname': 'Reference_Recurrent_Neural_Net',
            'ylabel': 'RNN (mean +/- std)'
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
            hplot.plot_time_series(df, f'{mat} {geometry}: Stromverlauf der Vorschubachse in x-Richtung',
                             f'Verlauf_{material}_{geometry}_Ref',
                             col_name='curr_x', y_configs=y_configs,)


