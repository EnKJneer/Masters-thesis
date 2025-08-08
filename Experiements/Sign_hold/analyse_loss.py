from collections import deque

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def sign_hold(v, eps = 1e-1, n = 5):
    # Initialisierung des Arrays z mit Nullen
    z = np.zeros(len(v))

    # Initialisierung des FiFo h mit Länge 5 und Initialwerten 0
    h = deque([1, 1, 1, 1, 1], maxlen=n)

    # Berechnung von z
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])

        if i >= n-1:  # Da wir ab dem 5. Element starten wollen
            # Berechne zi als Vorzeichen der Summe
            z[i] = np.sign(sum(h))

    return z
def plot_time_series(data, title, dpi=300, label = 'v_x', ylabel = 'v_x', ylabel2 = None, f_a = 50, align=False):
    """
    Erstellt einen Zeitverlaufsplan mit zwei y-Achsen.

    :param data: DataFrame mit den Daten
    :param filename: Dateiname zum Speichern des Plots
    :param title: Titel des Plots
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    """
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=dpi)
    if 'time' in data.columns:
        time = data['time']
    else:
        time = data.index * (1/f_a)
    ax1.plot(time, data[label], label=label, color='tab:green')
    ax1.set_xlabel('Time in s')
    ax1.set_ylabel(label)
    ax1.set_title(title)
    ax1.legend(loc='upper left')

    if ylabel is not None:
        # Zweite y-Achse für curr_x
        ax2 = ax1.twinx()
        line, = ax2.plot(time, data[ylabel], label=ylabel, color='tab:red')
        ax2.set_ylabel(ylabel)
        ax2.legend(loc='upper right')

        if ylabel2 is not None:
            line, = ax2.plot(time, data[ylabel2], label=ylabel2, color='tab:orange')

        if align:
            y1_min, y1_max = ax1.get_ylim()
            y2_min, y2_max = ax2.get_ylim()

            # Berechnung der Skalierungsfaktoren für die Achsen
            scaling_factor = y1_max / y2_max

            # Skalieren der zweiten Achse
            ax2.set_ylim(y2_min * scaling_factor, y2_max * scaling_factor)

            # Anpassen der zweiten Linie an die Skalierung
            line.set_ydata(data[ylabel] * scaling_factor)

    plt.show()

if __name__ == '__main__':
    path_data = 'Results/ST_Plate_Notch-2025_07_29_17_25_48/Predictions/S235JR_Gear_Normal_3.csv'

    data = pd.read_csv(path_data)

    data['z_x'] = sign_hold(data['v_x'])
    data['sign_curr_x'] = -np.sign(data['curr_x'])
    data['sign_diff'] = data['sign_curr_x'] - data['z_x']
    data['error'] = data['curr_x'] - data['ohne z_Recurrent_Neural_Net']

    plot_time_series(data, 'error', dpi=300, label = 'error')
    plot_time_series(data, 'error', dpi=300, label = 'error', ylabel = 'z_x', ylabel2 = 'sign_curr_x')
    plot_time_series(data, 'error', dpi=300, label='error', ylabel='sign_diff')
    plot_time_series(data, 'Recurrent_Neural_Net (optimiert)', dpi=300, label='curr_x', ylabel='error')

