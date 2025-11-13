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

def setup_plot(
    kit_dark_blue: str,
    line_size: float,
    fontsize_axis_label: int,
) -> tuple:
    """Erstellt die Grundstruktur der Plots (Achsen, Stile, etc.)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Stile für ax_v (Vorschubgeschwindigkeit)
    #ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(kit_dark_blue)
    ax.spines['bottom'].set_color(kit_dark_blue)
    ax.spines['left'].set_linewidth(line_size)
    ax.spines['bottom'].set_linewidth(line_size)

    # Plot Vorschubgeschwindigkeit
    ax.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='both', colors=kit_dark_blue, labelsize=fontsize_axis_label)

    return fig, ax


def add_axes_labels(
    ax_i: plt.Axes,
    label: str,
    kit_dark_blue: str,
    fontsize_axis_label: int,
    line_size: float,
    time
) -> None:
    """Fügt Achsenbeschriftungen hinzu."""
    # X-Achsenbeschriftung (ax_i)
    xmin = min(time)
    xmax = max(time)
    ymin, ymax = ax_i.get_ylim()
    x_pos = xmin
    y_pos = -0.3 * ymax
    arrow_length = 0.04 * (xmax - xmin)
    ax_i.set_xlim(left=min(x_pos, xmin), right=xmax + arrow_length/2)
    ax_i.annotate('', xy=(xmax + arrow_length/2, 0), xytext=(xmax - arrow_length/2, 0),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_i.text(xmax*0.95, y_pos, r'$t$ in s', ha='left', va='center', color=kit_dark_blue, fontsize=fontsize_axis_label)

    # Y-Achsenbeschriftung (ax_i)
    y_pos = ymax * 0.85
    arrow_length = 0.04*(ymax-ymin)
    ax_i.set_ylim(bottom=min(y_pos, ymin), top=ymax + arrow_length / 2)
    ax_i.annotate('', xy=(xmin, ymax + arrow_length/2), xytext=(xmin, ymax - arrow_length/2),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_i.text(x_pos -0.1*xmin, y_pos - 0.06*(ymax-ymin), label,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=fontsize_axis_label)

def plot_time_series_sections(
        data: pd.DataFrame,
        title: str,
        filename: str,
        dpi: int = 300,
        col_name: str = 'curr_x',
        axis_name: str = '$I$ in A',
        label: str = 'Strom-Messwerte',
        speed_threshold: float = 1.0,
        f_a: int = 50,
        path: str = 'Plots',
        y_configs: List[Dict[str, str]] = None,
        data_types=['.svg', '.pdf'],
        fontsize_axis: int=16,
        fontsize_axis_label: int = 16,
        fontsize_label: int = 14,
        fontsize_title: int = 18,
        line_size: int = 1.5,
        plot_line_size: int = 2,
        v_colname: str = 'v_x',
        v_threshold: float = 1.0,
) -> None:
    """Erstellt einen DIN 461 konformen Zeitverlaufsplan mit:
    - Oberem Plot: Vorschubgeschwindigkeit (eingefärbt nach Vorzeichen der Kraft)
    - Unterem Plot: Strom (eingefärbt nach z für |v| < speed_threshold)
    - Unterstützung für y_configs im unteren Plot
    """
    # Konfigurationen

    y_colors = [ hplottime.KIT_RED,  hplottime.KIT_ORANGE,  hplottime.KIT_MAGENTA,  hplottime.KIT_YELLOW,  hplottime.KIT_GREEN]

    # Zeitachse
    time = data.index / f_a

    # ----- Plot aufbauen (zwei Achsen: oben für Vorschubgeschwindigkeit, unten für Strom) -----
    fig, ax =  setup_plot(
        hplottime.KIT_DARK_BLUE, line_size, fontsize_axis
    )
    # Bereiche mit |v| < v_threshold markieren
    v_abs = np.abs(data[v_colname])
    low_speed_mask = v_abs < v_threshold
    diff = np.diff(low_speed_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if low_speed_mask.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if low_speed_mask.iloc[-1]:
        ends = np.append(ends, len(low_speed_mask) - 1)

    for start, end in zip(starts, ends):
        ax.axvspan(time[start], time[end], color= hplottime.KIT_GREEN, alpha=0.2, linewidth=0)

    # ----- Plot für Strom-Messwerte und y_configs -----
    lines_pred =  hplottime.plot_y_configs(ax, data, y_configs, y_colors, time, plot_line_size)
    line_i, = ax.plot(time, data[col_name], label=label, color= hplottime.KIT_BLUE, linewidth=plot_line_size)

    # ----- Achsenbeschriftungen -----
    add_axes_labels(ax, axis_name,  hplottime.KIT_DARK_BLUE, fontsize_axis_label, line_size, time)

    # ----- Legende -----
    legend_elements = lines_pred + [line_i]
    legend_labels = [line.get_label() for line in legend_elements]

    legend_elements.append(Patch(facecolor= hplottime.KIT_GREEN, alpha=0.2, label=f'Bereiche mit |v| < {v_threshold} m/s'))
    legend_labels.append(f'|v| < {v_threshold} m/s')



    # ----- Titel, Legende und Speichern -----
    hplottime.add_legend_and_save(
        fig, ax, legend_elements, legend_labels, title,  hplottime.KIT_DARK_BLUE, fontsize_title, fontsize_axis_label,
        filename, path, dpi, data_types
    )

if __name__ == '__main__':

    # Beispielaufruf der Funktion
    materials = ['AL2007T4'] #'S235JR',
    geometries = [ 'Gear'] #'Plate',

    paths = ['Predictions']

    y_configs = [
        {
            'ycolname': 'ST_Plate_Notch_Mixed_Experts_2',
            'ylabel': 'Experten Modell'
        },

    ]
    start = 50 * 15
    ende = 50 * 30

    for material in materials:

        for geometry in geometries:

            data = []
            for path in paths:

                file = f'DMC60H_{material}_{geometry}_Normal_3.csv'
                df = pd.read_csv(f'{path}/{file}')
                df = df.loc[df.index[start:ende]]
                data.append(df)

            df = pd.concat(data, axis=1)

            # Doppelte Spalten (identisch) entfernen
            df = df.loc[:, ~df.columns.duplicated()]
            if material == 'AL2007T4':
                mat = 'Aluminium'
            else:
                mat = 'Stahl'

            plot_time_series_sections(df, f'{mat} {geometry}:\nStromverlauf der Vorschubachse in x-Richtung',
                             f'Detail_Verlauf_{material}_{geometry}',
                             col_name='curr_x', dpi=600,
                             y_configs=y_configs, path='Plots_Thesis',
                                                fontsize_axis=22, fontsize_axis_label=24,
                                                fontsize_title=28,
                                                )


