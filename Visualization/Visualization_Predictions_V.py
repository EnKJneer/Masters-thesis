import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(data, title, filename, dpi=300,
                     col_name='v_x', axis='Abweichung', label='Geschwindigkeit in m/s',
                     col_name_right='Abweichung RF', axis_right='Abweichung', label_right='',
                     ycolname_1='Abweichung RF', ylabel_1='Abweichung RF',
                     ycolname_2='Abweichung RNN', ylabel_2='Abweichung RNN',
                     f_a=50, path='Plots'):
    import matplotlib.pyplot as plt
    import os
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_orange = "#FFC000"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_magenta = "#A3107C"
    time = data.index / f_a
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=dpi)

    # Linke y-Achse
    ax1.spines['left'].set_position('zero')
    ax1.spines['left'].set_color(kit_dark_blue)
    ax1.spines['left'].set_linewidth(1.0)
    ax1.spines['bottom'].set_position('zero')
    ax1.spines['bottom'].set_color(kit_dark_blue)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Rechte y-Achse
    ax2 = ax1.twinx()
    ax2.spines['right'].set_color(kit_dark_blue)
    ax2.spines['right'].set_linewidth(1.0)
    ax2.spines['bottom'].set_position('zero')
    ax2.spines['bottom'].set_color(kit_dark_blue)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # Hauptdaten
    line0, = ax1.plot(time, data[col_name], label=label, color=kit_blue, linewidth=2)
    line_right, = ax2.plot(time, data[col_name_right], label=label_right,
                           color=kit_magenta, linewidth=2, linestyle="--")

    # Symmetrische Achsen
    ymin, ymax = ax1.get_ylim()
    max_abs1 = max(abs(ymin), abs(ymax))
    ax1.set_ylim(-max_abs1, max_abs1)
    dx = (time[-1]) * 0.06
    ax1.set_xlim(time[0] - dx, time[-1] + dx)
    ymin2, ymax2 = ax2.get_ylim()
    max_abs2 = max(abs(ymin2), abs(ymax2))
    ax2.set_ylim(-max_abs2, max_abs2)

    # Zweite y-Achse (ax2) verschieben
    ax2.spines['right'].set_position(('outward', -dx*8))

    # Abweichungen mit Std
    def plot_prediction_with_std(ax, data, base_label, color, label=''):
        cols = [col for col in data.columns if col.startswith(base_label)]
        if not cols:
            return None, None
        mean = data[cols].mean(axis=1)
        std = data[cols].std(axis=1)
        line, = ax.plot(time, mean, label=label, color=color, linewidth=2)
        ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.2)
        return line, mean

    line1, mean1 = plot_prediction_with_std(ax1, data, ycolname_1, kit_red, ylabel_1)
    line2, mean2 = plot_prediction_with_std(ax1, data, ycolname_2, kit_orange, ylabel_2)

    # Grid
    ax1.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5)
    ax1.set_axisbelow(True)

    # Linke Y-Achse: Zahlen in kit_dark_blue
    ax1.tick_params(axis='y', colors=kit_dark_blue)
    # Rechte Y-Achse: Zahlen in kit_dark_blue
    ax2.tick_params(axis='y', colors=kit_dark_blue)
    # X-Achse: Zahlen in kit_dark_blue
    ax1.tick_params(axis='x', colors=kit_dark_blue)

    # Achsenpfeile
    xmin, xmax = ax1.get_xlim()

    # X-Achse Pfeil
    y_pos = -0.07 * max_abs1
    ax1.annotate('', xy=(xmax, y_pos), xytext=(xmax * 0.95, y_pos),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue,
                                 lw=1.5, mutation_scale=15),
                 clip_on=False)
    ax1.text(xmax * 0.95, 1.5*y_pos, r'$t$ in s',
             ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Linke Y-Achse Pfeil
    x_pos = -0.02 * (xmax - xmin)
    ax1.annotate('', xy=(x_pos, max_abs1 * 0.9), xytext=(x_pos, max_abs1 * 0.8),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue,
                                 lw=1.5, mutation_scale=15),
                 clip_on=False)
    ax1.text(-0.06 * (xmax - xmin), max_abs1 * 0.8, axis,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # Rechte Y-Achse Pfeil
    x_pos = xmax - 0.02 * (xmax - xmin)
    ax2.annotate('', xy=(x_pos, max_abs2 * 0.9), xytext=(x_pos, max_abs2 * 0.8),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue,
                                 lw=1.5, mutation_scale=15),
                 clip_on=False)
    ax2.text(xmax - 0.01 * (xmax - xmin), max_abs2 * 0.8, axis_right,
             ha='left', va='bottom', color=kit_dark_blue, fontsize=12)

    # Titel
    ax1.set_title(title, color=kit_dark_blue, fontsize=14, fontweight='bold', pad=20)

    # Legende verschieben
    lines = [line for line in [line0, line_right, line1, line2] if line is not None]
    labels = [line.get_label() for line in lines]
    legend = ax1.legend(lines, labels, loc='upper right',
                        frameon=True, facecolor='white',
                        edgecolor=kit_dark_blue, framealpha=1.0,
                        bbox_to_anchor=((time[-1] - dx)/time[-1], 1.0))
    legend.get_frame().set_linewidth(1.0)
    for text in legend.get_texts():
        text.set_color(kit_dark_blue)

    # Speichern
    plot_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', pad_inches=0.3, facecolor='white')
    plt.close(fig)
    print(f'saved as {plot_path}')

'''# Beispielaufruf der Funktion
material = 'AL_2007_T4'
geometry = 'Plate'
path_data = '..\\Experiements/Referenzmodelle/Results/Reference-2025_08_18_20_02_44/Predictions'
file = f'{material}_{geometry}_Normal_3.csv'
data = pd.read_csv(f'{path_data}/{file}')

plot_time_series(
    data,
    f'Aluminium {geometry}: Strom- und Geschwindigkeitsverlauf',
    f'Verlauf_{material}_{geometry}_Ref_RF_vx',
    col_name='curr_x',
    label='Strom in A',
    dpi=1200,
    ycolname_1='Reference_Random_Forest',
    ylabel_1='Random Forest',
    vx_col='v_x',
    vx_label='Geschwindigkeit in mm/s'
)'''

# Beispielaufruf der Funktion
material = 'S235JR'
geometry = 'Plate'

path_data = '..\\Experiements\\Hyperparameteroptimization/Results/Random_Forest/2025_07_28_14_40_41/Predictions'
file = f'{material}_Plate_Normal_3.csv'
df = pd.DataFrame()
data = pd.read_csv(f'{path_data}/{file}')
df['curr_x'] = data['curr_x']
df['Random Forest'] = data['ST_Plate_Notch_Random_Forest_RandomSampler']

path_data = '..\\Experiements\\Hyperparameteroptimization/Results/Recurrent_Neural_Net/2025_07_28_19_20_29/Predictions'
data = pd.read_csv(f'{path_data}/{file}')
df['Rekurrentes neuronales Netz'] = data['ST_Plate_Notch_Recurrent_Neural_Net_TPESampler']
df['v_x'] = data['v_x']

plot_time_series(
    df,
    f'Stahl {geometry}: Strom- und Geschwindigkeitsverlauf',
    f'Verlauf_{material}_{geometry}_RF_RNN',
    col_name='curr_x', axis='$I$ in A', label='Messung',
    dpi=1200,
    ycolname_1='Random Forest', ylabel_1='Random Forest',
    ycolname_2='Rekurrentes neuronales Netz', ylabel_2='Rekurrentes neuronales Netz',
    col_name_right='v_x', axis_right='$v$ in mm/s', label_right='Vorschubgeschwindigkeit'
)
