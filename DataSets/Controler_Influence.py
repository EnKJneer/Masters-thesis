import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_2d_with_color(x_values, y_values, color_values, filename, label='|v_x + v_y|', title = '2D Plot von pos_x und pos_y mit Farbe', dpi=300):
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
    for i in range(len(x_values) - 1):
        plt.plot(x_values[i:i+2], y_values[i:i+2], c=plt.cm.viridis(normalized_color_values[i]))

    # Erstellen eines Streudiagramms, um die Farbskala anzuzeigen
    sc = plt.scatter(x_values, y_values, c=color_values, cmap='viridis', s=5)

    # Hinzufügen einer Farbskala
    plt.colorbar(sc, label=label)

    # Beschriftungen und Titel hinzufügen
    plt.xlabel('pos_x')
    plt.ylabel('pos_y')
    plt.title(f'{title}')
    plt.savefig(filename + '.png')
    # Anzeigen des Plots
    #plt.show()
def plot_time_series(data, filename, title, dpi=300, label = 'a_x'):
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
    ax1.plot(data['Zeit'], data['CONT_DEV_X'], label='CONT_DEV_X', color='tab:green')
    #ax1.plot(data['Zeit'], data[label], label=label, color='tab:green')
    ax1.set_xlabel('Zeit')
    ax1.set_ylabel('Werte')
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax1.set_ylim(-0.002, 0.002)  # Beschränkung der ersten y-Achse auf +/- 0.1

    # Zweite y-Achse für curr_x
    ax2 = ax1.twinx()
    ax2.plot(data['Zeit'], data['CURRENT_X'], label='CURRENT_X', color='tab:red')
    ax2.set_ylabel('CURRENT_X')
    ax2.legend(loc='upper right')

    plt.show()

path_additional_data = 'AdditionalDataFiltered'
path_data = 'DataFiltered'
files = os.listdir(path_additional_data)
files = ['AL_2007_T4_Plate_Normal', 'S235JR_Plate_Normal']
for file in files:
    file = file.replace('.csv', '')
    data = pd.read_csv(f'{path_data}/{file}_3.csv')
    data_additional = pd.read_csv(f'{path_additional_data}/{file}.csv')

    # Fenstergröße für den gleitenden Mittelwert
    window_size = 5  # Annahme Daten mit 10kHz gesampelt

    # Gleitenden Mittelwert berechnen (optional: min_periods=1 damit am Anfang nicht NaN)
    data_smoothed = data_additional.rolling(window=window_size, min_periods=1).mean()

    # Heruntersampeln, z.B. jeden 10. Wert behalten
    data_reduced = data_smoothed.iloc[::window_size].reset_index(drop=True)

    print(data.columns)
    print(data_reduced.columns)
    print(data.shape)
    print(data_additional.shape)
    print(data_reduced.shape)

    x_values = data['pos_x']
    y_values = data['pos_y']
    color_values = np.abs(data['v_x'] + data['v_y'])
    #plot_2d_with_color(x_values, y_values, color_values)

    x_values = data_reduced['ENC_POS_X']
    y_values = data_reduced['ENC_POS_Y']
    #color_values = np.abs(data_reduced['CMD_SPEED_X'] + data_reduced['CMD_SPEED_Y'])
    key= 'CTRL_DIFF2'#'CONT_DEV'
    values = np.abs(data_reduced['CTRL_DIFF2_X'] + data_reduced['CTRL_DIFF2_Y'])
    #max_value = 0.1
    #values = np.where(values > max_value, max_value, values)
    color_values = values
    label = '|CTRL_DIFF2_X + CTRL_DIFF2_Y|'
    #plot_2d_with_color(x_values, y_values, color_values, f'Plots/{file}_{key}', label = label, title = file, dpi = 600)

    data_reduced['Zeit'] = np.arange(len(data_reduced))
    data['Zeit'] = np.arange(len(data))

    plot_time_series(data_reduced, f'Plots/{file}_time_series', title=f'{file}')
    #plot_time_series(data, f'Plots/{file}_time_series', title=f'{file}')