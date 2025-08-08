import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_time_series(data, title, filename, dpi=300, col_name = 'v_x', label='Geschwindigkeit in m/s', ylabel1='Abweichung RF', ylabel2='Abweichung RNN', f_a=50, path='Plots'):
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

    line0, = ax1.plot(time, data[col_name], label='Messwerte', color=kit_blue)
    ax1.set_xlabel('Zeit in s', color=kit_dark_blue)
    ax1.set_ylabel(label, color=kit_dark_blue)
    ax1.set_title(title, color=kit_dark_blue)
    ax1.tick_params(axis='x', colors=kit_dark_blue)
    ax1.tick_params(axis='y', colors=kit_dark_blue)

    # Zweite y-Achse für Abweichung RF und Abweichung RNN
    line1, = ax1.plot(time, data[ylabel1], label=ylabel1, color=kit_red)
    line2, = ax1.plot(time, data[ylabel2], label=ylabel2, color=kit_orange)


    # Kombinierte Legende für die zweite Achse
    lines = [line0, line1, line2]
    labels = [line0.get_label(), line1.get_label(), line2.get_label()]

    ax1.legend(lines, labels, loc='upper left')

    plot_path = os.path.join(path, filename)
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Beispielaufruf der Funktion
material = 'AL_2007_T4'
path_data = 'Hyperparameteroptimization/Results/Random_Forest/2025_07_28_14_40_41/Predictions'
file = f'{material}_Plate_Normal_3.csv'
df = pd.DataFrame()
data = pd.read_csv(f'{path_data}/{file}')
df['curr_x'] = data['curr_x']
df['Random Forest'] = data['ST_Plate_Notch_Random_Forest_RandomSampler']

path_data = 'Hyperparameteroptimization/Results/Recurrent_Neural_Net/2025_07_28_19_20_29/Predictions'
data = pd.read_csv(f'{path_data}/{file}')
df['Rekurrentes neuronales Netz'] = data['ST_Plate_Notch_Recurrent_Neural_Net_TPESampler']

plot_time_series(df, f'{material} Plate: Stromverlauf', f'Verlauf_{material}_Plate',
                 col_name = 'curr_x', label='Strom in A', dpi=1200,
                 ylabel1='Random Forest', ylabel2='Rekurrentes neuronales Netz')
