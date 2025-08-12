import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import variance
from xarray import align


def plot_time_series(data, title, filename, dpi=300, col_name='v_x', label='Messwerte',
                     label_axis='Geschwindigkeit in m/s',
                     col_name1=None, label1='Simulation', col_name2=None, label2='Simulation 2',
                     f_a=50, path='Plots'):
    """
    Erstellt einen DIN 461 konformen Zeitverlaufsplan.
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

    # DIN 461 konforme Figur erstellen (Seitenverhältnis ca. 3:2)
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

    # Plot der Hauptdaten
    line0, = ax1.plot(time, data[col_name], label=label, color=kit_blue, linewidth=1.5)

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
    x_label_pos = xmax# * 0.95
    y_label_pos = -0.08 * (ymax - ymin)

    ax1.annotate('', xy=(x_label_pos + arrow_length, y_label_pos),
                 xytext=(x_label_pos , y_label_pos),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue,
                                 lw=1.5, shrinkA=0, shrinkB=0,
                                 mutation_scale=15))

    ax1.text(x_label_pos - 0.06 * (xmax - xmin), y_label_pos, r'$t$ in s',
             ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Y-Achse: Pfeil bei der Beschriftung (oben zeigend)
    x_label_pos_y = -0.06 * (xmax - 0)
    y_label_pos_y = ymax * 0.85

    ax1.annotate('', xy=(x_label_pos_y, y_label_pos_y + arrow_height),
                 xytext=(x_label_pos_y, y_label_pos_y),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue,
                                 lw=1.5, shrinkA=0, shrinkB=0,
                                 mutation_scale=15))

    ax1.text(x_label_pos_y, y_label_pos_y - 0.04 * (ymax - ymin), label_axis,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # Titel mit DIN 461 konformer Positionierung
    ax1.set_title(title, color=kit_dark_blue, fontsize=14, fontweight='bold', pad=20)

    # Legende vorläufige Liste erstellen
    lines = [line0]
    labels = [line0.get_label()]

    # Weitere Datenreihen hinzufügen, falls vorhanden
    if col_name1 is not None:
        line1, = ax1.plot(time, data[col_name1], label=label1, color=kit_red, linewidth=1.5)
        lines.append(line1)
        labels.append(line1.get_label())

        if col_name2 is not None:
            line2, = ax1.plot(time, data[col_name2], label=label2, color=kit_orange, linewidth=1.5)
            lines.append(line2)
            labels.append(line2.get_label())

    # DIN 461: Legende mit Rahmen und korrekter Positionierung (oben rechts)
    legend = ax1.legend(lines, labels, loc='upper right',
                        frameon=True, fancybox=False, shadow=False,
                        framealpha=1.0, facecolor='white', edgecolor=kit_dark_blue)
    legend.get_frame().set_linewidth(1.0)

    # Schriftfarbe der Legende
    for text in legend.get_texts():
        text.set_color(kit_dark_blue)

    # DIN 461: Achsenbegrenzungen anpassen, damit Nullpunkt sichtbar ist
    ax1.set_xlim(left=min(x_label_pos_y, xmin), right=xmax * 1.05)
    ax1.set_ylim(bottom=min(y_label_pos, ymin), top=ymax * 1.05)

    # Speicherpfad erstellen und Plot speichern
    plot_path = os.path.join(path, filename)
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')

# Beispielaufruf der Funktion
material = 'AL_2007_T4'
geometry = 'Gear'
variant = 'Normal'
version = '3'
path_data = '..\\DataSets\\Data' #'..\\Archiv\\DataSets\\OldData_Aligned'
file = f'{material}_{geometry}_{variant}_{version}.csv'

data = pd.read_csv(f'{path_data}/{file}')
data['f_x'] = -200*data['f_x']

plot_time_series(data, f'{material} {geometry} {variant}: Verlauf der Prozesskraft', f'Kräfte_{material}_{geometry}_{variant}_new',
                 col_name = 'f_x', label='Messwerte', label_axis='$F$ in N', dpi=1200,
                 col_name1='f_x_sim', label1='Simulation')
