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

def plot_time_series(
    data: pd.DataFrame,
    title: str,
    filename: str,
    dpi: int = 300,
    col_name: str = 'curr_x',
    label: str = '$I$\nin A',
    y_configs: List[Dict[str, str]] = None,
    f_a: int = 50,
    path: str = 'Plots'
) -> None:
    """
    Erstellt einen DIN 461 konformen Zeitverlaufsplan mit:
    - Oberem Plot: Vorschubgeschwindigkeit (v_x)
    - Unterem Plot: Stromvorhersage mit farblicher Kennzeichnung von Bereichen mit |v_x| < 1 m/s

    Args:
        data: DataFrame mit den Daten
        title: Titel des Plots
        filename: Dateiname zum Speichern des Plots
        dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
        col_name: Spaltenname für Strom-Messwerte
        label: Beschriftung der y-Achse für Strom
        y_configs: Liste von Dictionaries mit 'ycolname' und 'ylabel' für zusätzliche y-Achsen
                  Beispiel: [{'ycolname': 'Abweichung_RF', 'ylabel': 'Random Forest'},
                             {'ycolname': 'Abweichung_RNN', 'ylabel': 'RNN'}]
        f_a: Abtastfrequenz in Hz (Standard: 50)
        path: Pfad zum Speichern der Plots
    """
    # KIT-Farben
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_orange = "#FFC000"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_magenta = "#A3107C"
    kit_gray = "#767676"

    # Standardfarben für zusätzliche y-Achsen (kann erweitert werden)
    y_colors = [kit_red, kit_orange, kit_magenta, kit_yellow, kit_green]

    time = data.index / f_a

    # Erstelle Figure mit zwei Achsen
    fig, ax_i = plt.subplots(figsize=(12, 8), dpi=dpi)


    # ----- Unterer Plot (Stromvorhersage) -----
    ax_i.spines['left'].set_position('zero')
    ax_i.spines['bottom'].set_position('zero')
    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)
    ax_i.spines['left'].set_color(kit_dark_blue)
    ax_i.spines['bottom'].set_color(kit_dark_blue)
    ax_i.spines['left'].set_linewidth(1.0)
    ax_i.spines['bottom'].set_linewidth(1.0)

    # Plot für Vorhersagen (dynamisch basierend auf y_configs)
    lines_pred = []
    if y_configs is None:
        y_configs = []

    for i, config in enumerate(y_configs):
        color = y_colors[i % len(y_colors)]
        line, _ = plot_prediction_with_std(
            data,
            config['ycolname'],
            color,
            config['ylabel']
        )
        if line is not None:
            lines_pred.append(line)

    # Grid und Achsenbeschriftung
    ax_i.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5)
    ax_i.set_axisbelow(True)
    ax_i.tick_params(axis='both', colors=kit_dark_blue)

    # X-Achsenbeschriftung (nur beim unteren Plot)
    xmin, xmax = ax_i.get_xlim()
    ymin, ymax = ax_i.get_ylim()
    y_pos = -0.07 * ymax

    ax_i.annotate('', xy=(xmax, 0), xytext=(xmax*0.95, 0),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5))
    ax_i.text(xmax*0.95, 1.5*y_pos, r'$t$ in s',
             ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Y-Achsenbeschriftung
    x_pos = -0.06 * (xmax - xmin)
    y_pos = ymax * 0.85
    ax_i.annotate('', xy=(0, ymax),
                 xytext=(0, ymax - 0.04*(ymax-ymin)),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5))
    ax_i.text(x_pos, y_pos - 0.04*(ymax-ymin), label,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # Titel über beiden Plots
    fig.suptitle(title, color=kit_dark_blue, fontsize=14,
                fontweight='bold', y=0.98)

    # Plot für Strom-Messwerte
    line_i, = ax_i.plot(time, data[col_name], label='Messwerte',
                       color=kit_blue, linewidth=2)

    '''    
    # Kombinierte Legende
    legend_elements = [line_i] + lines_pred
    legend_labels = [line.get_label() for line in legend_elements if line is not None]

    fig.legend(
        handles=legend_elements,
        labels=legend_labels,
        loc='lower center',
        ncol=3,
        frameon=True,
        facecolor='white',
        edgecolor=kit_dark_blue,
        framealpha=1.0,
        bbox_to_anchor=(0.5, -0.05)
    )'''
    # Kombinierte Legende für die Achsen
    legend_elements = [line_i] + lines_pred
    legend_labels = [line.get_label() for line in legend_elements if line is not None]
    lines = [line for line in legend_elements if line is not None]
    labels = [line.get_label() for line in lines if line is not None]
    legend = ax_i.legend(lines, legend_labels, loc='upper right',
                        frameon=True, fancybox=False, shadow=False,
                        framealpha=1.0, facecolor='white', edgecolor=kit_dark_blue)
    legend.get_frame().set_linewidth(1.0)
    for text in legend.get_texts():
        text.set_color(kit_dark_blue)

    # Achsenbegrenzungen anpassen
    ax_i.set_xlim(left=min(x_pos, xmin), right=xmax*1.05)
    #ax_i.set_ylim(bottom=min(y_pos, ymin), top=ymax*1.05)

    # Speichern des Plots
    plot_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(plot_path + '.pdf', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved as {plot_path}')

def plot_prediction_with_std(data, base_label, color, label=''):
    """
    Hilfsfunktion zum Plotten von Vorhersagen mit Standardabweichung.
    """
    cols = [col for col in data.columns if col.startswith(base_label)]
    if not cols:
        return None, None

    mean = data[cols].mean(axis=1)
    std = data[cols].std(axis=1)
    line, = plt.gca().plot(data.index / 50, mean, label=label, color=color, linewidth=2)
    plt.gca().fill_between(data.index / 50, mean - std, mean + std, color=color, alpha=0.2)
    return line, mean

if __name__ == '__main__':

    # Beispielaufruf der Funktion
    materials = ['S235JR', 'AL2007T4']
    geometries = ['Plate', 'Gear']

    paths = ['Predictions']

    y_configs = [
        {
            'ycolname': 'ST_Plate_Notch_Friction_Model',
            'ylabel': 'Empirisches Modell'
        },
        {
            'ycolname': 'ST_Plate_Notch_Hybrid_RNN',
            'ylabel': 'Residual-Modell'
        }
    ]
    '''        {
        'ycolname': 'ST_Plate_Notch_Recurrent_Neural_Net',
        'ylabel': 'RNN'
    },'''

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
                             f'Verlauf_{material}_{geometry}',
                             col_name='curr_x', label='$I$\nin A', dpi=600,
                             y_configs=y_configs, path='Plots_Thesis')


