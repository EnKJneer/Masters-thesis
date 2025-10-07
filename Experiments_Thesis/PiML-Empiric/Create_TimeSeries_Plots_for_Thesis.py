import copy
import json
import os
import ast
import re
from collections import deque

import shap
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
    label: str = 'Strom in A',
    v_colname: str = 'v_x',
    v_label: str = 'Vorschubgeschwindigkeit',
    v_axis: str = 'v in m/s',
    f_x_sim_col: str = 'f_x_sim',
    f_x_sim_label: str = 'Simulierte Kraft',
    f_x_sim_axis: str = '$F$\nin N',
    z_col: str = 'z',
    speed_threshold: float = 1.0,
    f_a: int = 50,
    path: str = 'Plots',
    lane1: float = 130.0,
    lane2: float = 310.0,
) -> None:
    """
    Erstellt einen DIN 461 konformen Zeitverlaufsplan mit:
    - Oberem Plot: Vorschubgeschwindigkeit (v_x)
    - Unterem Plot: Strommesswerte + simulierte Kraft (f_x_sim) auf zweiter y-Achse
    - Farbige Bereiche NUR für |v| < speed_threshold (rot/orange)
    - Gestrichelte Näherungslinien bei ±lane1 und ±^lane2
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

    time = data.index / f_a
    fig, (ax_v, ax_i) = plt.subplots(
        2, 1, figsize=(12, 10), dpi=dpi,
        sharex=True, height_ratios=[1, 2],
        gridspec_kw={'hspace': 0.05}
    )

    # ----- Berechnung der Bereiche mit |v| < speed_threshold -----
    v_abs = np.abs(data[v_colname])
    low_speed_mask = v_abs < speed_threshold
    diff = np.diff(low_speed_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if low_speed_mask.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if low_speed_mask.iloc[-1]:
        ends = np.append(ends, len(low_speed_mask)-1)

    # ----- Einfärben der Bereiche mit |v| < speed_threshold, abhängig von z -----
    alpha = 0.2
    for start, end in zip(starts, ends):
        z_segment = data[z_col].iloc[start:end+1]
        if (z_segment > 0).any():
            color = kit_red
        else:
            color = kit_orange
        ax_i.axvspan(time[start], time[end], color=color, alpha=alpha, linewidth=0)

    # ----- Oberer Plot (Vorschubgeschwindigkeit) -----
    ax_v.spines['left'].set_position('zero')
    ax_v.spines['bottom'].set_position('zero')
    ax_v.spines['top'].set_visible(False)
    ax_v.spines['right'].set_visible(False)
    ax_v.spines['left'].set_color(kit_dark_blue)
    ax_v.spines['bottom'].set_color(kit_dark_blue)
    ax_v.spines['left'].set_linewidth(1.0)
    ax_v.spines['bottom'].set_linewidth(1.0)

    line_v, = ax_v.plot(time, data[v_colname], label=v_label, color=kit_blue, linewidth=2)
    ax_v.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5)
    ax_v.set_axisbelow(True)
    ax_v.tick_params(axis='both', colors=kit_dark_blue)

    # Achsenbeschriftung (Oberer Plot)
    xmin, xmax = ax_v.get_xlim()
    ymin, ymax = ax_v.get_ylim()
    y_pos = -0.07 * ymax
    ax_v.annotate('', xy=(xmax, 0), xytext=(xmax*0.95, 0),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5))
    ax_v.text(xmax*0.95, y_pos, r'$t$ in s', ha='left', va='center', color=kit_dark_blue, fontsize=12)

    x_label_pos_y = -0.06 * (xmax - xmin)
    y_label_pos_y = ymax * 0.65
    ax_v.annotate('', xy=(0, ymax), xytext=(0, ymax - 0.08*(ymax-ymin)),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5))
    ax_v.text(x_label_pos_y, y_label_pos_y - 0.04*(ymax-ymin), v_axis,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # ----- Unterer Plot (Strom + f_x_sim) -----
    ax_i.spines['left'].set_position('zero')
    ax_i.spines['bottom'].set_position('zero')
    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)
    ax_i.spines['left'].set_color(kit_dark_blue)
    ax_i.spines['bottom'].set_color(kit_dark_blue)
    ax_i.spines['left'].set_linewidth(1.0)
    ax_i.spines['bottom'].set_linewidth(1.0)

    # Plot Strom-Messwerte (primäre y-Achse)
    line_i, = ax_i.plot(time, data[col_name], label='Strom-Messwerte', color=kit_dark_blue, linewidth=2)

    # Zweite y-Achse für f_x_sim
    ax_i2 = ax_i.twinx()
    ax_i2.spines['right'].set_position(('axes', 1.0))
    ax_i2.spines['right'].set_color(kit_blue)
    ax_i2.spines['right'].set_linewidth(1.0)
    ax_i2.spines['top'].set_visible(False)
    ax_i2.spines['left'].set_visible(False)
    ax_i2.tick_params(axis='y', colors=kit_blue)

    # Plot f_x_sim
    line_f, = ax_i2.plot(time, data[f_x_sim_col], label=f_x_sim_label, color=kit_blue, linewidth=2)

    # ----- Symmetrische Achsengrenzen berechnen -----
    ymin_i, ymax_i = ax_i.get_ylim()
    max_abs_i = max(abs(ymin_i), abs(ymax_i))
    ax_i.set_ylim(-max_abs_i * 1.05, max_abs_i * 1.05)

    ymin_f, ymax_f = ax_i2.get_ylim()
    max_abs_f = max(abs(ymin_f), abs(ymax_f))
    ax_i2.set_ylim(-max_abs_f * 1.05, max_abs_f * 1.05)

    # ----- Gestrichelte Näherungslinien (+/-lane1 und +/-310) -----
    # Für Strom (ax_i)
    ax_i2.axhline(y=lane1, color=kit_green, linestyle='--', linewidth=2, label=f'Näherung +{lane1} F')
    ax_i2.axhline(y=-lane1, color=kit_green, linestyle='--', linewidth=2, label=f'Näherung -{lane1} F')
    ax_i2.axhline(y=lane2, color=kit_magenta, linestyle='--', linewidth=2, label=f'Näherung +{lane2} F')
    ax_i2.axhline(y=-lane2, color=kit_magenta, linestyle='--', linewidth=2, label=f'Näherung -{lane2} F')

    # Grid und Achsenbeschriftung
    ax_i.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5)
    ax_i.set_axisbelow(True)
    ax_i.tick_params(axis='both', colors=kit_dark_blue)

    # X-Achsenbeschriftung
    xmin, xmax = ax_i.get_xlim()
    ymin, ymax = ax_i.get_ylim()
    y_pos = -0.07 * ymax
    ax_i.annotate('', xy=(xmax, 0), xytext=(xmax*0.95, 0),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5))
    ax_i.text(xmax*0.95, 1.5*y_pos, r'$t$ in s',
             ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Y-Achsenbeschriftung (Strom)
    x_pos = -0.06 * (xmax - xmin)
    y_pos = ymax * 0.85
    ax_i.annotate('', xy=(0, ymax*1.05), xytext=(0, ymax*1.05 - 0.04*(ymax-ymin)),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5))
    ax_i.text(x_pos, y_pos - 0.04*(ymax-ymin), label,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # Y-Achsenbeschriftung für f_x_sim
    ymin_f, ymax_f = ax_i2.get_ylim()
    ax_i2.annotate('', xy=(xmax*1.05, ymax_f*1.05), xytext=(xmax*1.05, ymax_f*1.05 - 0.04*(ymax_f-ymin_f)),
                  arrowprops=dict(arrowstyle='->', color=kit_blue, lw=1.5))
    ax_i2.text(xmax, ymax_f * 0.85, f_x_sim_axis,
              ha='left', va='center', color=kit_blue, fontsize=12)

    # Titel
    fig.suptitle(title, color=kit_dark_blue, fontsize=14, fontweight='bold', y=0.98)

    # Legende (inkl. Näherungslinien)
    legend_elements = [line_i, line_v, line_f]
    legend_labels = [line.get_label() for line in legend_elements]

    # Farbige Bereiche zur Legende hinzufügen

    legend_elements.extend([
        Patch(facecolor=kit_red, alpha=alpha, label=f'Bereiche mit |v| < {speed_threshold} und z > 0'),
        Patch(facecolor=kit_orange, alpha=alpha, label=f'Bereiche mit |v| < {speed_threshold} und z ≤ 0'),
    ])
    legend_labels.extend([
        f'|v| < {speed_threshold}, z > 0',
        f'|v| < {speed_threshold}, z ≤ 0'
    ])

    # Gestrichelte Linien zur Legende hinzufügen

    legend_elements.extend([
        Line2D([0], [0], color=kit_green, linestyle='--', label=f'Näherung ±{lane1} N'),
        Line2D([0], [0], color=kit_magenta, linestyle='--', label=f'Näherung ±{lane2} N'),
    ])
    legend_labels.extend([f'±{lane1} N', f'±{lane2} N'])

    fig.legend(
        handles=legend_elements,
        labels=legend_labels,
        loc='lower center',
        ncol=4,  # 4 Spalten für bessere Lesbarkeit
        frameon=True,
        facecolor='white',
        edgecolor=kit_dark_blue,
        framealpha=1.0,
        bbox_to_anchor=(0.5, -0.05)
    )

    # Achsenbegrenzungen
    ax_i.set_xlim(left=min(x_pos, xmin), right=xmax*1.05)
    ax_i.set_ylim(bottom=min(y_pos, ymin), top=ymax*1.05)
    ax_i2.set_ylim(bottom=ymin_f, top=ymax_f * 1.05)
    ax_v.set_xlim(left=min(x_pos, xmin), right=xmax*1.05)

    # Speichern
    plot_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
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

def sign_hold(v, eps=1e-1, n=3, init=-1):
    # Initialisierung des Arrays z mit Nullen
    z = np.zeros(len(v))
    h_init = np.ones(n) * init

    assert n > 1

    # Initialisierung des FiFo h mit Länge 5 und Initialwerten 0
    h = deque(h_init, maxlen=n)

    # Berechnung von z
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])

        if i >= n - 1:  # Da wir ab dem 5. Element starten wollen
            # Berechne zi als Vorzeichen der Summe
            z[i] = np.sign(sum(h))

    return z

if __name__ == '__main__':
    # Beispielaufruf mit der neuen y_configs-Struktur
    materials = ['S235JR'] #, 'AL2007T4'
    geometries = ['Plate'] #, 'Gear'
    paths = [
        'Results/EmpiricModel-2025_10_07_09_23_22/Predictions',
    ]

    for material in materials:
        for geometry in geometries:
            data = []
            for path in paths:
                file = f'DMC60H_{material}_{geometry}_Normal_3.csv'
                data.append(pd.read_csv(f'{path}/{file}'))

            df = pd.concat(data, axis=1).reset_index(drop=True)
            df = df.loc[:, ~df.columns.duplicated()]
            df['f_x_sim'] = -df['f_x_sim']
            df['z'] = sign_hold(df['v_x'])

            mat = 'Aluminium' if material == 'AL2007T4' else 'Stahl'

            plot_time_series(
                df,
                f'{mat} {geometry}: Einflüsse auf den Stromverlauf',
                f'Verlauf_{material}_{geometry}_mit_Vorschub.pdf',
                col_name='curr_x',
                label='$I$\nin A',
                v_colname='v_x',
                v_label='Vorschubgeschwindigkeit',
                v_axis='$v$\nin m/s',
                dpi=600
            )
