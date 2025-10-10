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

def plot_time_series(data, title, filename, dpi=300, col_name='v_x',
                     label='Geschwindigkeit in m/s',
                     ycolname_1='Abweichung RF', ylabel_1 = 'Abweichung RF',
                     ycolname_2='Abweichung RNN', ylabel_2 = 'Abweichung RF', f_a=50, path='Plots'):
    """
    Erstellt einen DIN 461 konformen Zeitverlaufsplan mit zwei y-Achsen.
    :param data: DataFrame mit den Daten
    :param filename: Dateiname zum Speichern des Plots
    :param title: Titel des Plots
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    """
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_orange = "#FFC000"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_magenta = "#A3107C"

    time = data.index / f_a
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=dpi)

    # DIN 461: Achsen müssen durch den Nullpunkt gehen
    ax1.spines['left'].set_position('zero')
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(kit_dark_blue)
    ax1.spines['bottom'].set_color(kit_dark_blue)
    ax1.spines['left'].set_linewidth(1.0)
    ax1.spines['bottom'].set_linewidth(1.0)

    # Umgang mit mehreren Vorhersagen (z. B. ylabel1_seed_1, ylabel1_seed_2, ...)
    def plot_prediction_with_std(data, base_label, color, label=''):
        # Suche alle Spalten, die mit base_label beginnen
        cols = [col for col in data.columns if col.startswith(base_label)]
        if not cols:
            return None, None
        # Berechne Mittelwert und Standardabweichung
        mean = data[cols].mean(axis=1)
        std = data[cols].std(axis=1)
        # Plot Mittelwert
        line, = ax1.plot(time, mean, label=label, color=color, linewidth=2)
        # Plot Standardabweichung als schattierten Bereich
        ax1.fill_between(time, mean - std, mean + std, color=color, alpha=0.2)
        return line, mean

    # Plot für ylabel1 (z. B. Abweichung RF)
    line1, mean1 = plot_prediction_with_std(data, ycolname_1, kit_red, ylabel_1)
    # Plot für ylabel2 (z. B. Abweichung RNN)
    line2, mean2 = plot_prediction_with_std(data, ycolname_2, kit_orange, ylabel_2)

    # Plot der Hauptdaten
    line0, = ax1.plot(time, data[col_name], label='Messwerte', color=kit_blue, linewidth=2)

    # DIN 461: Beschriftungen in kit_dark_blue
    ax1.tick_params(axis='x', colors=kit_dark_blue, direction='inout', length=6)
    ax1.tick_params(axis='y', colors=kit_dark_blue, direction='inout', length=6)

    # Grid nach DIN 461
    ax1.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5, linestyle='-')
    ax1.set_axisbelow(True)

    # Achsenbeschriftungen mit Pfeilen
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    arrow_length = 0.03 * (xmax - xmin)
    arrow_height = 0.04 * (ymax - ymin)

    # X-Achse: Pfeil bei der Beschriftung
    x_label_pos = xmax
    y_label_pos = -0.08 * (ymax - ymin)
    ax1.annotate('', xy=(x_label_pos + arrow_length, y_label_pos),
                 xytext=(x_label_pos, y_label_pos),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue,
                                 lw=1.5, shrinkA=0, shrinkB=0,
                                 mutation_scale=15))
    ax1.text(x_label_pos - 0.06 * (xmax - xmin), y_label_pos, r'$t$ in s',
             ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Y-Achse: Pfeil bei der Beschriftung
    x_label_pos_y = -0.06 * (xmax - 0)
    y_label_pos_y = ymax * 0.85
    ax1.annotate('', xy=(x_label_pos_y, y_label_pos_y + arrow_height),
                 xytext=(x_label_pos_y, y_label_pos_y),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue,
                                 lw=1.5, shrinkA=0, shrinkB=0,
                                 mutation_scale=15))
    ax1.text(x_label_pos_y, y_label_pos_y - 0.04 * (ymax - ymin), label,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # Titel mit DIN 461 konformer Positionierung
    ax1.set_title(title, color=kit_dark_blue, fontsize=14, fontweight='bold', pad=20)

    # Kombinierte Legende für die Achsen
    lines = [line for line in [line0, line1, line2] if line is not None]
    labels = [line.get_label() for line in lines if line is not None]
    legend = ax1.legend(lines, labels, loc='upper right',
                        frameon=True, fancybox=False, shadow=False,
                        framealpha=1.0, facecolor='white', edgecolor=kit_dark_blue)
    legend.get_frame().set_linewidth(1.0)
    for text in legend.get_texts():
        text.set_color(kit_dark_blue)

    # DIN 461: Achsenbegrenzungen anpassen
    ax1.set_xlim(left=min(x_label_pos_y, xmin), right=xmax * 1.05)
    ax1.set_ylim(bottom=min(y_label_pos, ymin), top=ymax * 1.05)

    # Speichern des Plots
    plot_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(plot_path + '.pdf', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved as {plot_path}')

if __name__ == '__main__':

    # Beispielaufruf der Funktion
    materials = ['S235JR', 'AL2007T4']
    geometries = ['Plate', 'Gear']

    paths = ['Results/Recurrent_Neural_Net-2025_10_02_16_20_24/Predictions',
             'Results/Random_Forest-2025_09_16_15_19_42/Predictions']

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
            plot_time_series(df, f'{mat} {geometry}: Stromverlauf der Vorschubachse in x-Richtung',
                             f'Verlauf_{material}_{geometry}_Ref_RF_opt',
                             col_name='curr_x', label='Strom in A', dpi=600,
                             ycolname_1='ST_Plate_Notch_Random_Forest_GridSampler', ylabel_1='Random Forest',
                             ycolname_2='ST_Plate_Notch_Recurrent_Neural_Net_GridSampler', ylabel_2='RNN')


