import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_time_series(data, title, filename, dpi=300, label='v_x', ylabel1='Abweichung RF', ylabel2=None, align=False, f_a=50, path='Plots'):
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

    ax1.plot(time, data[label], label=label, color=kit_blue)
    ax1.set_xlabel('Zeit in s', color=kit_dark_blue)
    ax1.set_ylabel(label, color=kit_dark_blue)
    ax1.set_title(title, color=kit_dark_blue)
    ax1.tick_params(axis='x', colors=kit_dark_blue)
    ax1.tick_params(axis='y', colors=kit_dark_blue)
    ax1.legend(loc='upper left')

    # Zweite y-Achse für Abweichung RF und Abweichung RNN
    if ylabel1 is not None:
        ax2 = ax1.twinx()
        line1, = ax2.plot(time, data[ylabel1], label=ylabel1, color=kit_red)
        if ylabel2 is not None:
            line2, = ax2.plot(time, data[ylabel2], label=ylabel2, color=kit_orange)
            # Kombinierte Legende für die zweite Achse
            lines = [line1, line2]
            labels = [line1.get_label(), line2.get_label()]
        else:
            # Kombinierte Legende für die zweite Achse
            lines = [line1]
            labels = [line1.get_label()]
        ax2.set_ylabel(ylabel1, color=kit_dark_blue)
        ax2.tick_params(axis='y', colors=kit_dark_blue)


        ax2.legend(lines, labels, loc='upper right')

        if align:
            y1_min, y1_max = ax1.get_ylim()
            y2_min, y2_max = ax2.get_ylim()
            scaling_factor = y1_max / y2_max
            ax2.set_ylim(y2_min * scaling_factor, y2_max * scaling_factor)
            line1.set_ydata(data[ylabel1] * scaling_factor)
            if ylabel2 is not None:
                line2.set_ydata(data[ylabel2] * scaling_factor)

    plot_path = os.path.join(path, filename)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Beispielaufruf der Funktion
material = 'AL_2007_T4'
path_data = '..\\DataSets\\Data'#'..\\Archiv\\DataSets\\Data'
file = f'{material}_Plate_Normal_3.csv'

data = pd.read_csv(f'{path_data}/{file}')
data['f_x'] = -200*data['f_x']
plot_time_series(data, f'{material} Plate', f'Kräfte_{material}_Plate', label='f_x_sim', dpi=1200, ylabel1='f_x', align=True)