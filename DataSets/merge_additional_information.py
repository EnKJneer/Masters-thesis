import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_with_color(x_values, y_values, color_values, name='|v_x + v_y|', dpi=300):
    """
    Erstellt einen 2D-Plot mit Linien, deren Farbe basierend auf den color_values bestimmt wird.

    :param x_values: Liste oder Array der x-Werte
    :param y_values: Liste oder Array der y-Werte
    :param color_values: Liste oder Array der Werte, die die Farbe bestimmen
    :param name: Name der Farbskala (Standard: '|v_x + v_y|')
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
    plt.colorbar(sc, label=name)

    # Beschriftungen und Titel hinzufügen
    plt.xlabel('pos_x')
    plt.ylabel('pos_y')
    plt.title(f'2D Plot von pos_x und pos_y mit Farbe basierend auf {name}')

    # Anzeigen des Plots
    plt.show()

path_additional_data = 'AdditionalDataFiltered'
path_data = 'DataFiltered'

file = 'AL_2007_T4_Plate_SF'
data = pd.read_csv(f'{path_data}/{file}_1.csv')
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
plot_2d_with_color(x_values, y_values, color_values)

x_values = data_reduced['ENC_POS_X']
y_values = data_reduced['ENC_POS_Y']
#color_values = np.abs(data_reduced['CMD_SPEED_X'] + data_reduced['CMD_SPEED_Y'])
values = np.abs(data_reduced['CONT_DEV_X'] + data_reduced['CONT_DEV_Y'])
max_value = 0.1
values = np.where(values > max_value, max_value, values)
color_values = values
plot_2d_with_color(x_values, y_values, color_values, name = '|CONT_DEV_X + CONT_DEV_Y|', dpi = 600)