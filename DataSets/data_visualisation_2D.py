import os
from collections import deque

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def plot_time_series(data, title, dpi=300, label = 'v_x'):
    """
    Erstellt einen Zeitverlaufsplot mit zwei y-Achsen.

    :param data: DataFrame mit den Daten
    :param filename: Dateiname zum Speichern des Plots
    :param title: Titel des Plots
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    """
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=dpi)

    # Plot der CTRL_DIFF, CTRL_DIFF2 und Cont_DEV_X Daten
    #ax1.plot(data['Zeit'], data['CTRL_DIFF_X'], label='CTRL_DIFF_X', color='tab:blue')
    #ax1.plot(data['Zeit'], data['CTRL_DIFF2_X'], label='CTRL_DIFF2_X', color='tab:orange')
    #ax1.plot(data['Zeit'], data['CONT_DEV_X'], label='CONT_DEV_X', color='tab:green')
    ax1.plot(data.index, data[label], label=label, color='tab:green')
    ax1.set_xlabel('Zeit')
    ax1.set_ylabel('Werte')
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    #ax1.set_ylim(-0.000001, 0.000001)

    # Zweite y-Achse für curr_x
    ax2 = ax1.twinx()
    ax2.plot(data.index, data['curr_x'], label='curr_x', color='tab:red')
    ax2.set_ylabel('curr_x')
    ax2.legend(loc='upper right')
    #ax2.set_ylim(-2, 2)

    plt.show()

path_data = 'DataFiltered'

files = os.listdir(path_data)
files = ['AL_2007_T4_Plate_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']
files = ['AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv']
for file in files:
    #file = file.replace('.csv', '')
    data = pd.read_csv(f'{path_data}/{file}')

    print(data.columns)
    print(data.shape)

    data['z_x'] = sign_hold(data['v_x'])
    data['z_y'] = sign_hold(data['v_y'])
    data['z_sp'] = -data['f_sp_sim']
    data['z_mrr'] = sign_hold(data['v_x']) * data['materialremoved_sim']
    xlabel = 'pos_x'
    ylabel = 'pos_y'
    label = 'f_y_sim'
    n = 25
    data = data.iloc[:-n]
    x_values = data[xlabel]
    y_values = data[ylabel]
    color_values = data[label] * (data['v_y']) / (np.abs(data['v_x']) + np.abs(data['v_y']))
    max_value = 2#-3 for curr_y # 2 for curr_x
    min_value = -2#-7 for curr_y # -2 for curr_x
    #color_values = np.clip(color_values, min_value, max_value)

    name = file.replace('.csv', '')
    #plot_2d_with_color(x_values, y_values, color_values, f'Plots/{name}_{xlabel}_{label}', label = label, title = file, dpi = 600, xlabel = xlabel, ylabel = ylabel)
    #plot_time_series(data, name, label='f_x_sim', dpi=300)
    #plot_time_series(data, name, label='f_y_sim', dpi=300)
    plot_time_series(data, name, label='v_x', dpi=300)
    plot_time_series(data, name, label='pos_x', dpi=300)
    #plot_time_series(data, name, label='f_z_sim', dpi=300)
    #plot_time_series(data, name, label='z_sp', dpi=300)
    #plot_time_series(data, name, label='z_mrr', dpi=300)
    #plot_time_series(data, name, label='v_x', dpi=300)
    #plot_time_series(data, name, label='z_x', dpi=300)