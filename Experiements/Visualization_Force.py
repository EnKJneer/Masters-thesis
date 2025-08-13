import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import variance
from xarray import align


def plot_time_series(data, title, filename, dpi=300, col_name = 'v_x', label = 'Messwerte',
                     label_axis='Geschwindigkeit in m/s',
                     col_name1=None, label1 = 'Simulation', col_name2=None, label2 = 'Simulation 2',
                     f_a=50, path='Plots'):
    """
    Erstellt einen Zeitverlaufsplan mit zwei y-Achsen.
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
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=dpi)

    line0, = ax1.plot(time, data[col_name], label=label, color=kit_blue)
    ax1.set_xlabel('Zeit in s', color=kit_dark_blue)
    ax1.set_ylabel(label_axis, color=kit_dark_blue)
    ax1.set_title(title, color=kit_dark_blue)
    ax1.tick_params(axis='x', colors=kit_dark_blue)
    ax1.tick_params(axis='y', colors=kit_dark_blue)

    lines = [line0]
    labels = [line0.get_label()]
    if col_name1 is not None:

        # Zweite y-Achse für Abweichung RF und Abweichung RNN
        line1, = ax1.plot(time, data[col_name1], label=label1, color=kit_red)

        # Kombinierte Legende für die zweite Achse
        lines.append(line1)
        labels.append(line1.get_label())

        if col_name2 is not None:
            line2, = ax1.plot(time, data[col_name2], label=label2, color=kit_orange)

            # Kombinierte Legende für die zweite Achse
            lines.append(line2)
            labels.append(line2.get_label())

    ax1.legend(lines, labels, loc='upper left')

    plot_path = os.path.join(path, filename)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Beispielaufruf der Funktion
material = 'S235JR'
geometry = 'Notch'
variant = 'Depth'
version = '3'
path_data = '..\\DataSets\\Data' #'..\\Archiv\\DataSets\\OldData_Aligned'
file = f'{material}_{geometry}_{variant}_{version}.csv'

data = pd.read_csv(f'{path_data}/{file}')
data['f_x'] = -200*data['f_x']

plot_time_series(data, f'{material} {geometry} {variant}: Stromverlauf', f'Kräfte_{material}_{geometry}_{variant}',
                 col_name = 'f_x', label='Messwerte', label_axis='Kraft in N', dpi=1200,
                 col_name1='f_x_sim', label1='Simulation')
