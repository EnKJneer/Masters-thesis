import os
from collections import deque

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt


def sign_hold(v, eps = 1e-1):
    # Initialisierung des Arrays z mit Nullen
    z = np.zeros(len(v))

    # Initialisierung des FiFo h mit Länge 5 und Initialwerten 0
    h = deque([1, 1, 1, 1, 1], maxlen=5)

    # Berechnung von z
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])

        if i >= 4:  # Da wir ab dem 5. Element starten wollen
            # Berechne zi als Vorzeichen der Summe
            z[i] = np.sign(sum(h))

    return z

def plot_2d_with_color(x_values, y_values, color_values, label='|v_x + v_y|', title = '2D Plot von pos_x und pos_y mit Farbe', dpi=300, xlabel = 'pos_x', ylabel = 'pos_y'):
    """
    Erstellt einen 2D-Plot mit Linien, deren Farbe basierend auf den color_values bestimmt wird.

    :param x_values: Liste oder Array der x-Werte
    :param y_values: Liste oder Array der y-Werte
    :param color_values: Liste oder Array der Werte, die die Farbe bestimmen
    :param label: Name der Farbskala (Standard: '|v_x + v_y|')
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    """
    # Erstellen des Plots mit höherer Auflösung
    plt.figure(figsize=(10, 6), dpi=dpi)

    # Normalisieren der color_values für den Farbverlauf
    normalized_color_values = (color_values - np.min(color_values)) / (np.max(color_values) - np.min(color_values))

    # Erstellen eines Farbverlaufs basierend auf den color_values
    #for i in range(len(x_values) - 1):
    #    plt.plot(x_values[i:i+2], y_values[i:i+2], c=plt.cm.viridis(normalized_color_values[i]))

    # Erstellen eines Streudiagramms, um die Farbskala anzuzeigen
    sc = plt.scatter(x_values, y_values, c=color_values, cmap='viridis', s=1)

    # Hinzufügen einer Farbskala
    plt.colorbar(sc, label=label)

    # Beschriftungen und Titel hinzufügen
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title}')
    #plt.savefig(filename + '.png')
    # Anzeigen des Plots
    plt.show()

def plot_time_series(data, title, dpi=300, label='v_x', ylabel='curr_x', f_a=50, align_axis=False):
    """
    Erstellt einen Zeitverlaufsplan mit zwei y-Achsen.

    :param data: DataFrame mit den Daten
    :param title: Titel des Plots
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    :param label: Bezeichnung der ersten y-Achse
    :param ylabel: Bezeichnung der zweiten y-Achse
    :param f_a: Abtastrate
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
        ax2.plot(time, data[ylabel], label=ylabel, color='tab:red', linestyle='--', alpha=0.8)
        ax2.set_ylabel(ylabel)
        ax2.legend(loc='upper right')

        if align_axis:
            # Berechne den maximalen absoluten Wert für jede Achse separat
            y1_min, y1_max = ax1.get_ylim()
            y2_min, y2_max = ax2.get_ylim()

            abs_max1 = max(abs(y1_min), abs(y1_max))
            abs_max2 = max(abs(y2_min), abs(y2_max))

            ax1.set_ylim(-abs_max1, abs_max1)
            ax2.set_ylim(-abs_max2, abs_max2)

    plt.show()

def butter_lowpass(cutoff, order, nyq_freq=0.5):
    normal_cutoff = cutoff / nyq_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, order):
    b, a = butter_lowpass(cutoff, order)
    data2 = data.copy()
    for col in data.columns:
        data2[col] = filtfilt(b, a, data2[col])
    return data2


files = ['..\\DataSetsV3/Data2/S235JR_Plate_Normal_2.csv']
for file in files:
    #file = file.replace('.csv', '')
    data = pd.read_csv(file)

    name = file.replace('.csv', '')
    #t_e = data.index[-1] * 1/500
    #print(t_e)
    data['f_x'] = -data['f_x']
    data['v_z'] = np.clip(data['v_z'], -1, 1)
    plot_time_series(data, name, label='f_x_sim', dpi=300, ylabel='f_x', align_axis=True)

    data['materialremoved_sim_norm'] = (
            (data['materialremoved_sim'] - data['materialremoved_sim'].min()) /
            (data['materialremoved_sim'].max() - data['materialremoved_sim'].min())
    )

    data['test'] = data['materialremoved_sim_norm'] * data['f_x_sim']
    plot_time_series(data, name, label='test', dpi=300, ylabel='f_x', align_axis=True)

    data['v_f'] = (data['v_x']**2 + data['v_y']**2)**0.5
    plot_time_series(data, name, label='v_f', dpi=300, ylabel=None, align_axis=True)