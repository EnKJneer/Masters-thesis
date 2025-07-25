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

def plot_2d_with_color(x_values, y_values, color_values, filename, label='|v_x + v_y|', title = '2D Plot von pos_x und pos_y mit Farbe', dpi=300, xlabel = 'pos_x', ylabel = 'pos_y'):
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

def plot_time_series(data, title, dpi=300, label = 'v_x', ylabel = 'curr_x', f_a = 50):
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
        ax2.plot(time, data[ylabel], label=ylabel, color='tab:red')
        ax2.set_ylabel(ylabel)
        ax2.legend(loc='upper right')

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

path_data = 'DataSimulated'
files = os.listdir(path_data)

files = ['S235JR_Plate_Depth.csv']
for file in files:
    #file = file.replace('.csv', '')
    data = pd.read_csv(f'{path_data}/{file}')
    cutoff = 0.1
    filter_order = 4
    #data = apply_lowpass_filter(data, cutoff, filter_order)
    #n = int(len(data)*1.1/3)
    #data = data.iloc[:2*n, :]
    #print(data.columns)
    #print(data.shape)

    #color_values = data['v_x']
    #max_value = 2#-3 for curr_y # 2 for curr_x
    #min_value = -2#-7 for curr_y # -2 for curr_x
    #color_values = np.clip(color_values, min_value, max_value)

    name = file.replace('.csv', '')
    #t_e = data.index[-1] * 1/500
    #print(t_e)
    data['f_x'] = -data['f_x']*200
    data['f_y'] = data['f_y']*200
    plot_time_series(data, name, label='f_x_sim', dpi=300, ylabel='f_x')
    plot_time_series(data, name, label='f_y_sim', dpi=300, ylabel='f_y')
    #plot_time_series(data, name, label='f_y', dpi=300, ylabel='curr_y')
    #plot_time_series(data, name, label='f_x', dpi=300, ylabel='curr_x')
    #plot_time_series(data, name, label='v_sp', dpi=300, ylabel='curr_sp')
    #plot_time_series(data, name, label='v_x', dpi=300, ylabel='curr_x')
    #plot_time_series(data, name, label='v_y', dpi=300, ylabel='curr_y')
    #plot_time_series(data, name, label='v_z', dpi=300, ylabel='curr_z')
    plot_time_series(data, name, label='f_x_sim', dpi=300, ylabel='pos_x')
    plot_time_series(data, name, label='f_x_sim', dpi=300, ylabel='pos_y')
    plot_time_series(data, name, label='f_x_sim', dpi=300, ylabel='materialremoved_sim')
    diff = data['f_x'] - data['f_x_sim']
    plot_2d_with_color(data['pos_x'], data['pos_y'], diff, f'{name}_diff.png', label='diff')
    plot_2d_with_color(data['pos_x'], data['pos_y'], data['materialremoved_sim'], f'{name}_diff.png', label='materialremoved_sim')
