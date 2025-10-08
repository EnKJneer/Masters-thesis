import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_time_series_with_variance(data_list, title, dpi=300, col_name='curr_x', label_axis='$I$ in A', f_a=50,
                                   filename = None, path = 'Plots'):
    """
    Erstellt einen DIN 461 konformen Zeitverlaufsplot mit Mittelwert, Standardabweichung und Originalkurven.
    :param data_list: Liste von DataFrames (jeweils eine Version)
    :param title: Titel des Plots
    :param dpi: Auflösung des Plots
    :param col_name: Spaltenname der zu plottenden Daten (z.B. 'curr_x')
    :param label_axis: Beschriftung der y-Achse
    :param f_a: Abtastrate für die Zeitachse
    """
    # Farben des KIT
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_orange = "#FFC000"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_magenta = "#A3107C"

    # Minimale Länge der Datenreihen bestimmen
    min_length = min(len(df[col_name]) for df in data_list)

    # Daten auf minimale Länge kürzen
    truncated_data = [df[col_name][:min_length] for df in data_list]

    # Zeitachse erstellen (basierend auf der minimalen Länge)
    time = np.arange(min_length) / f_a

    # Mittelwert und Standardabweichung berechnen
    mean_values = np.mean(truncated_data, axis=0)
    std_values = np.std(truncated_data, axis=0)

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)

    # DIN 461: Achsen durch Nullpunkt
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(kit_dark_blue)
    ax.spines['bottom'].set_color(kit_dark_blue)

    colors = [kit_red, kit_orange, kit_magenta]  # Farben für Version 1, 2, 3
    # Originalkurven (transparent) plotten
    for i, values in enumerate(truncated_data):
        ax.plot(time, values, color=colors[i], alpha=0.3, linewidth=1, label=f'Version {i + 1}')

    # Mittelwert plotten
    line_mean, = ax.plot(time, mean_values, label='Mittelwert', color=kit_blue, linewidth=2)

    # Standardabweichung als Schatten darstellen
    ax.fill_between(time, mean_values - std_values, mean_values + std_values,
                    color=kit_blue, alpha=0.2, label='±1 Std.-Abw.')

    # DIN 461: Grid und Achsenbeschriftung
    ax.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5, linestyle='-')
    ax.tick_params(axis='x', colors=kit_dark_blue, direction='inout', length=6)
    ax.tick_params(axis='y', colors=kit_dark_blue, direction='inout', length=6)

    # Achsenbeschriftung mit Pfeilen
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    arrow_length = 0.03 * (xmax - xmin)
    arrow_height = 0.04 * (ymax - ymin)

    # X-Achse: Pfeil
    x_label_pos = xmax
    y_label_pos = -0.08 * (ymax - ymin)
    ax.annotate('', xy=(x_label_pos + arrow_length, y_label_pos),
                xytext=(x_label_pos, y_label_pos),
                arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5))
    ax.text(x_label_pos - 0.06 * (xmax - xmin), y_label_pos, r'$t$ in s',
            ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Y-Achse: Pfeil
    x_label_pos_y = -0.06 * (xmax - 0)
    y_label_pos_y = ymax * 0.85
    ax.annotate('', xy=(x_label_pos_y, y_label_pos_y + arrow_height),
                xytext=(x_label_pos_y, y_label_pos_y),
                arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5))
    ax.text(x_label_pos_y, y_label_pos_y - 0.04 * (ymax - ymin), label_axis,
            ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # Titel
    ax.set_title(title, color=kit_dark_blue, fontsize=14, fontweight='bold', pad=20)

    # Legende
    ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False,
              framealpha=1.0, facecolor='white', edgecolor=kit_dark_blue)

    # Achsenbegrenzungen anpassen
    ax.set_xlim(left=min(x_label_pos_y, xmin), right=xmax * 1.05)
    ax.set_ylim(bottom=min(y_label_pos, ymin), top=ymax * 1.05)

    # Speichern oder anzeigen
    if filename:
        plot_path = os.path.join(path, filename)
        os.makedirs(path, exist_ok=True)
        fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        fig.savefig(plot_path +'.pdf', dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'Saved as {plot_path}')
    else:
        plt.show()

# Beispielaufruf
material = 'S235JR'
geometry = 'Plate'
variance = 'Normal'
path_data = '..\\DataSets_DMC60H_Plate_Notch_Gear\\Data_1'

# Daten für alle Versionen laden
data_list = []
for version in [1, 2, 3]:
    file = f'DMC60H_{material}_{geometry}_{variance}_{version}.csv'
    data = pd.read_csv(f'{path_data}/{file}')
    data_list.append(data)

if material == 'S235JR':
    mat = 'Stahl'
else:
    mat = 'Aluminium'
# Plot erstellen
plot_time_series_with_variance(
    data_list,
    f'{mat} {geometry}: Varianz des Motorstroms (Versionen 1-3)',
    col_name='curr_x',
    label_axis='$I$ in A',
    dpi=600,
    filename='Variance_' + file.replace('.csv', '')
)
