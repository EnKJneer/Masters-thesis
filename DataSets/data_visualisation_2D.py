import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    plt.savefig(filename + '.png')
    # Anzeigen des Plots
    #plt.show()

path_data = 'DataFiltered'

files = os.listdir(path_data)
files = ['AL_2007_T4_Gear_Normal_3.csv']
for file in files:
    if '_3' in file:
        #file = file.replace('.csv', '')
        data = pd.read_csv(f'{path_data}/{file}')

        print(data.columns)
        print(data.shape)

        xlabel = 'pos_x'
        ylabel = 'pos_y'
        label = 'curr_x'
        n = 400
        x_values = data[xlabel].iloc[:n]
        y_values = data[ylabel].iloc[:n]
        color_values = data[label].iloc[:n]
        max_value = 2#-3 for curr_y # 2 for curr_x
        min_value = -2#-7 for curr_y # -2 for curr_x
        color_values = np.clip(color_values, min_value, max_value)

        name = file.replace('.csv', '')
        plot_2d_with_color(x_values, y_values, color_values, f'Plots/{name}_{xlabel}_{label}_halb', label = label, title = file, dpi = 600, xlabel = xlabel, ylabel = ylabel)