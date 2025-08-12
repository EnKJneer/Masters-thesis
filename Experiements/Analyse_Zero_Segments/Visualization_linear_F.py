import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import os
from collections import deque
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("TkAgg")

def sign_hold(v, eps=1e0, n=3):
    e = np.zeros(n)
    z = np.zeros(len(v))
    h = deque(e, maxlen=n)
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])
        if i >= n-1:
            z[i] = np.sign(sum(h))
    return z

def hold(v, eps=1e0, n=23):
    e = np.zeros(n)
    z = np.zeros(len(v))
    h = deque(e, maxlen=n)
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])
        if i >= n-1:
            z[i] = np.mean(h)
    return z

def plot_time_series(data, title, x_col_name, y_col_name, f_col_name, x_label='Zeit in s', y_label='Messwerte', dpi=300):
    """
    Erstellt einen Zeitverlaufsplot mit einer x-Achse und einer y-Achse.

    :param data: DataFrame mit den Daten
    :param title: Titel des Plots
    :param x_col_name: Spaltenname der x-Achsen-Daten
    :param y_col_name: Spaltenname der y-Achsen-Daten
    :param f_col_name: Spaltenname der Daten, die die Farbe bestimmen
    :param x_label: Beschriftung der x-Achse
    :param y_label: Beschriftung der y-Achse
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    """
    # KIT-Farben definieren
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_dark_blue = "#002D4C"

    # Benutzerdefinierte Farbpalette erstellen (grün=gut, gelb=mittel, rot=schlecht)
    custom_cmap = LinearSegmentedColormap.from_list(
        'kit_colors',
        [kit_green, kit_yellow, kit_red],
        N=256
    )

    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=dpi)

    # DIN 461: Achsen müssen durch den Nullpunkt gehen
    ax1.spines['left'].set_position('zero')
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # DIN 461: Achsen in kit_dark_blue
    ax1.spines['left'].set_color(kit_dark_blue)
    ax1.spines['bottom'].set_color(kit_dark_blue)
    ax1.spines['left'].set_linewidth(1.0)
    ax1.spines['bottom'].set_linewidth(1.0)

    # Normalisierung der f_x_sim Werte für die Farbskala
    norm = plt.Normalize(data[f_col_name].min(), data[f_col_name].max())

    # Plot der Hauptdaten mit Farbverlauf basierend auf f_x_sim
    for i in range(len(data) - 1):
        ax1.plot(data[x_col_name][i:i+2], data[y_col_name][i:i+2],
                 color=custom_cmap(norm(data[f_col_name][i])), linewidth=1.5)

    # DIN 461: Beschriftungen in kit_dark_blue
    ax1.tick_params(axis='x', colors=kit_dark_blue, direction='inout', length=6)
    ax1.tick_params(axis='y', colors=kit_dark_blue, direction='inout', length=6)

    # Grid nach DIN 461 (optional, aber empfohlen)
    ax1.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5, linestyle='-')
    ax1.set_axisbelow(True)

    # Achsenbeschriftungen mit Pfeilen bei der Beschriftung
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()

    # Pfeillängen für Beschriftung
    arrow_length = 0.03 * (xmax - xmin)
    arrow_height = 0.04 * (ymax - ymin)

    # X-Achse: Pfeil bei der Beschriftung (rechts zeigend)
    x_label_pos = xmax
    y_label_pos = -0.08 * (ymax - ymin)
    ax1.annotate('', xy=(x_label_pos + arrow_length, y_label_pos),
                 xytext=(x_label_pos, y_label_pos),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue,
                                 lw=1.5, shrinkA=0, shrinkB=0,
                                 mutation_scale=15))
    ax1.text(x_label_pos - 0.06 * (xmax - xmin), y_label_pos, x_label,
             ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Y-Achse: Pfeil bei der Beschriftung (oben zeigend)
    x_label_pos_y = -0.06 * (xmax - 0)
    y_label_pos_y = ymax * 0.85
    ax1.annotate('', xy=(x_label_pos_y, y_label_pos_y + arrow_height),
                 xytext=(x_label_pos_y, y_label_pos_y),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue,
                                 lw=1.5, shrinkA=0, shrinkB=0,
                                 mutation_scale=15))
    ax1.text(x_label_pos_y, y_label_pos_y - 0.04 * (ymax - ymin), y_label,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # Titel mit DIN 461 konformer Positionierung
    ax1.set_title(title, color=kit_dark_blue, fontsize=14, fontweight='bold', pad=20)

    # DIN 461: Achsenbegrenzungen anpassen, damit Nullpunkt sichtbar ist
    ax1.set_xlim(left=min(x_label_pos_y, xmin), right=xmax * 1.05)
    ax1.set_ylim(bottom=min(y_label_pos, ymin), top=ymax * 1.05)

    plt.show()

# Beispielaufruf der Funktion
material = 'AL_2007_T4'
geometry = 'Plate'
versions = ['Normal'] # Beispielhafte Liste von Versionen
path_data = '../../DataSets/DataSimulated'

# Leerer DataFrame zur Sammlung aller Daten
all_data = pd.DataFrame()

for version in versions:
    file = f'{material}_{geometry}_{version}.csv'
    data = pd.read_csv(os.path.join(path_data, file))

    # Berechnungen für jede Version
    data['z_x'] = sign_hold(data['v_x'])
    data['t_x'] = hold(data['v_x'])

    # Daten zum gemeinsamen DataFrame hinzufügen
    all_data = pd.concat([all_data, data], ignore_index=True)

# Maskierung und Gradientberechnung für den gesamten Datensatz
mask = (abs(all_data['v_x']) < 1) & (all_data['z_x'] == 1)
all_data = all_data[mask].reset_index(drop=True)
all_data['time'] = all_data.index / 50

# Beispielaufruf der Funktion
plot_time_series(all_data, f'{material} {geometry}',
                 x_col_name='time', y_col_name='curr_x', f_col_name='pos_x',
                 x_label='t', y_label='I', dpi=600)
