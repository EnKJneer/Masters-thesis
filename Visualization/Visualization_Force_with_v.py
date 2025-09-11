import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import variance
from xarray import align

def plot_time_series(
    data,
    title,
    filename,
    dpi=300,
    col_name='v_x',
    label='Messwerte',
    label_axis='Geschwindigkeit in m/s',
    col_name1=None,
    label1='Simulation',
    col_name2=None,
    label2='Simulation 2',
    spatle_col_name='spatle_v_x',
    spatle_label='Spatel-Geschwindigkeit',
    spatle_label_axis='Spatel-Geschwindigkeit in m/s',
    f_a=50,
    path='Plots'
):
    """
    Erstellt einen DIN 461 konformen Zeitverlaufsplan mit einer zweiten y-Achse für die Spatel-Geschwindigkeit.
    :param data: DataFrame mit den Daten
    :param filename: Dateiname zum Speichern des Plots
    :param title: Titel des Plots
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    :param spatle_col_name: Spaltenname für die Spatel-Geschwindigkeit (Standard: 'spatle_v_x')
    :param spatle_label: Beschriftung für die Spatel-Geschwindigkeit (Standard: 'Spatel-Geschwindigkeit')
    :param spatle_label_axis: Beschriftung der zweiten y-Achse (Standard: 'Spatel-Geschwindigkeit in m/s')
    """
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_orange = "#FFC000"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_magenta = "#A3107C"

    time = data.index / f_a

    # DIN 461 konforme Figur erstellen (Seitenverhältnis ca. 3:2)
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=dpi)

    # DIN 461: Achsen müssen durch den Nullpunkt gehen
    ax1.spines['left'].set_position('zero')
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # DIN 461: Achsen in kit_dark_blue
    ax1.spines['left'].set_color(kit_dark_blue)
    ax1.spines['bottom'].set_color(kit_dark_blue)
    ax1.spines['left'].set_linewidth(1.0)
    ax1.spines['bottom'].set_linewidth(1.0)

    # Plot der Hauptdaten
    line0 = ax1.plot(time, data[col_name], label=label, color=kit_blue)[0]

    # DIN 461: Beschriftungen in kit_dark_blue
    ax1.tick_params(axis='x', colors=kit_dark_blue, direction='inout', length=6)
    ax1.tick_params(axis='y', colors=kit_dark_blue, direction='inout', length=6)

    # Grid nach DIN 461 (optional, aber empfohlen)
    ax1.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5, linestyle='-')
    ax1.set_axisbelow(True)

    # Achsenbeschriftungen mit Pfeilen bei der Beschriftung
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()

    # Pfeillängen für Beschriftung
    arrow_length = 0.03 * (xmax - xmin)
    arrow_height = 0.04 * (ymax - ymin)

    # X-Achse: Pfeil bei der Beschriftung (rechts zeigend)
    x_label_pos = xmax * 0.98
    y_label_pos = -0.08 * (ymax - ymin)
    ax1.annotate('', xy=(x_label_pos, -0.04 * (ymax - ymin)),
                 xytext=(x_label_pos - arrow_length, -0.04 * (ymax - ymin)),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue,
                                 lw=1.5, shrinkA=0, shrinkB=0,
                                 mutation_scale=15))
    ax1.text(x_label_pos - 0.04 * (xmax - xmin), y_label_pos, r'$t$ in s',
             ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Y-Achse: Pfeil bei der Beschriftung (oben zeigend)
    x_label_pos_y = -0.04 * (xmax - 0)
    y_label_pos_y = ymax * 0.82
    ax1.annotate('', xy=(x_label_pos_y, y_label_pos_y + arrow_height),
                 xytext=(x_label_pos_y, y_label_pos_y),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue,
                                 lw=1.5, shrinkA=0, shrinkB=0,
                                 mutation_scale=15))
    ax1.text(x_label_pos_y, y_label_pos_y - 0.04 * (ymax - ymin), label_axis,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # Titel mit DIN 461 konformer Positionierung
    ax1.set_title(title, color=kit_dark_blue, fontsize=14, fontweight='bold', pad=20)

    # Legende vorläufige Liste erstellen
    lines = [line0]
    labels = [line0.get_label()]

    # Weitere Datenreihen hinzufügen, falls vorhanden
    if col_name1 is not None:
        line1 = ax1.plot(time, data[col_name1], label=label1, color=kit_red)[0]
        lines.append(line1)
        labels.append(line1.get_label())
        if col_name2 is not None:
            line2 = ax1.plot(time, data[col_name2], label=label2, color=kit_orange)[0]
            lines.append(line2)
            labels.append(line2.get_label())

    # Zweite y-Achse für die Spatel-Geschwindigkeit
    ax2 = ax1.twinx()
    line_spatle = ax2.plot(time, data[spatle_col_name], label=spatle_label, color=kit_magenta)[0]
    lines.append(line_spatle)
    labels.append(line_spatle.get_label())

    # DIN 461: Beschriftungen für die zweite y-Achse
    ax2.tick_params(axis='y', colors=kit_magenta, direction='inout', length=6)
    ax2.spines['right'].set_color(kit_magenta)
    ax2.spines['right'].set_linewidth(1.0)

    # Pfeil für die zweite y-Achse (oben zeigend)
    x_label_pos_spatle = xmax * 1.02
    ymin_spatle, ymax_spatle = ax2.get_ylim()
    y_label_pos_spatle = ymax_spatle * 0.82
    ax2.annotate('', xy=(x_label_pos_spatle, y_label_pos_spatle + arrow_height),
                 xytext=(x_label_pos_spatle, y_label_pos_spatle),
                 arrowprops=dict(arrowstyle='->', color=kit_magenta,
                                 lw=1.5, shrinkA=0, shrinkB=0,
                                 mutation_scale=15))
    ax2.text(x_label_pos_spatle, y_label_pos_spatle - 0.04 * (ymax_spatle - ymin_spatle), spatle_label_axis,
             ha='left', va='bottom', color=kit_magenta, fontsize=12)

    # DIN 461: Legende mit Rahmen und korrekter Positionierung (oben rechts)
    legend = ax1.legend(lines, labels, loc='upper right',
                        frameon=True, fancybox=False, shadow=False,
                        framealpha=1.0, facecolor='white', edgecolor=kit_dark_blue)
    legend.get_frame().set_linewidth(1.0)

    # Schriftfarbe der Legende
    for text in legend.get_texts():
        text.set_color(kit_dark_blue)

    # Speicherpfad erstellen und Plot speichern
    plot_path = os.path.join(path, filename)
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.show()

# Beispielaufruf der Funktion
material = 'AL2007T4'
geometry = 'Plate'
variant = 'Normal'
version = '3'
parameter = '' #_Reference_Hanlin
path_data = f'..\\DataSets_CMX_Plate_Notch_Gear{parameter}/Data'
file = f'DMC60H_{material}_{geometry}_{variant}_{version}.csv' #
data = pd.read_csv(f'{path_data}/{file}')
data['f_x'] = -200*data['f_x'] # Bei Referenz 150 sonst 200

if material == 'S235JR':
    mat = 'Stahl'
else:
    mat = 'Aluminium'

plot_time_series(
    data,
    f'{mat} {geometry} {variant}: Verlauf der Prozesskraft in x-Richtung',
    f'Kräfte_{material}_{geometry}_{variant}{parameter}_v',
    col_name='f_x',
    label='Messwerte',
    label_axis='$F$ in N',
    dpi=300,
    col_name1='f_x_sim',
    label1='Simulation',
    spatle_col_name='v_x',
    spatle_label='Geschwindigkeit'
)
