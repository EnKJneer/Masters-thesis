import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_time_series(data, title, filename, dpi=300,
                     col_name='v_x', label='Messwerte',  label_axis='Geschwindigkeit in m/s',
                     col_name1=None, label1='Simulation',
                     col_name2=None, label2='Simulation 2', align=False, f_a=50, path='Plots'):
    """
    Erstellt einen DIN 461 konformen Zeitverlaufsplan mit zwei y-Achsen.
    :param data: DataFrame mit den Daten
    :param filename: Dateiname zum Speichern des Plots
    :param title: Titel des Plots
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    """
    # Farben des KIT (Karlsruher Institut für Technologie)
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_orange = "#FFC000"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_magenta = "#A3107C"

    time = data.index / f_a
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
    line0, = ax1.plot(time, data[col_name], label=label, color=kit_blue, linewidth=1.5)

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
    x_label_pos = xmax
    y_label_pos = -0.08 * (ymax - ymin)
    ax1.annotate('', xy=(x_label_pos + arrow_length, y_label_pos),
                 xytext=(x_label_pos, y_label_pos),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue,
                                 lw=1.5, shrinkA=0, shrinkB=0,
                                 mutation_scale=15))
    ax1.text(x_label_pos - 0.06 * (xmax - xmin), y_label_pos, r'$t$ in s',
             ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Y-Achse: Pfeil bei der Beschriftung (oben zeigend)
    x_label_pos_y = -0.06 * (xmax - 0)
    y_label_pos_y = ymax * 0.85
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
        line1, = ax1.plot(time, data[col_name1], label=label1, color=kit_red, linewidth=1.5)
        lines.append(line1)
        labels.append(line1.get_label())
        if col_name2 is not None:
            line2, = ax1.plot(time, data[col_name2], label=label2, color=kit_orange, linewidth=1.5)
            lines.append(line2)
            labels.append(line2.get_label())

    # DIN 461: Legende mit Rahmen und korrekter Positionierung (oben rechts)
    legend = ax1.legend(lines, labels, loc='upper right',
                        frameon=True, fancybox=False, shadow=False,
                        framealpha=1.0, facecolor='white', edgecolor=kit_dark_blue)
    legend.get_frame().set_linewidth(1.0)

    # Schriftfarbe der Legende
    for text in legend.get_texts():
        text.set_color(kit_dark_blue)

    # Zweite y-Achse für Abweichung RF und Abweichung RNN
    if col_name1 is not None:
        ax2 = ax1.twinx()
        line1, = ax2.plot(time, data[col_name1], label=label1, color=kit_red, linewidth=1.5)
        if col_name2 is not None:
            line2, = ax2.plot(time, data[col_name2], label=label2, color=kit_orange, linewidth=1.5)
            # Kombinierte Legende für die zweite Achse
            lines = [line1, line2]
            labels = [line1.get_label(), line2.get_label()]

            if align:
                y1_min, y1_max = ax1.get_ylim()
                y2_min, y2_max = ax2.get_ylim()
                scaling_factor = y1_max / y2_max
                ax2.set_ylim(y2_min * scaling_factor, y2_max * scaling_factor)
                line1.set_ydata(data[col_name2] * scaling_factor)
                line2.set_ydata(data[col_name2] * scaling_factor)

        else:
            # Kombinierte Legende für die zweite Achse
            lines = [line1]
            labels = [line1.get_label()]

        ax2.set_ylabel(label1, color=kit_dark_blue, fontsize=10)
        ax2.tick_params(axis='y', colors=kit_dark_blue, labelsize=10)
        ax2.legend(lines, labels, loc='upper right', fontsize=10, frameon=True, fancybox=False, shadow=False,
                   framealpha=1.0, facecolor='white', edgecolor=kit_dark_blue)

    # DIN 461: Achsenbegrenzungen anpassen, damit Nullpunkt sichtbar ist
    ax1.set_xlim(left=min(x_label_pos_y, xmin), right=xmax * 1.05)
    ax1.set_ylim(bottom=min(y_label_pos, ymin), top=ymax * 1.05)

    plot_path = os.path.join(path, filename)
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# Beispielaufruf der Funktion
material = 'S235JR'
geometry =  'Plate'
version = 'Normal'
path_data = '..\\DataSets_CMX_Plate_Notch_Gear/Data'#'..\\Archiv\\DataSets\\Data' #
file = f'DMC60H_{material}_{geometry}_{version}_3.csv'

data = pd.read_csv(f'{path_data}/{file}')
data['v'] = (data['v_x']**2 + data['v_y']**2)**0.5 *60
file = file.replace('.csv','')
#data['f_x'] = -200*data['f_x']
plot_time_series(data, f'{material} {geometry} {version}: Verlauf der Vorschubgeschwindigkeit',
                 f'V_{file}', col_name= 'v', label_axis ='$v$ in mm/min', label='Vorschubgeschwindigkeit', dpi=600)