import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def plot_time_series(
    data, title, filename, dpi=300, col_name='curr_x', label='Messwerte', axis_name='$I$ in A',
    ycolname_1='Abweichung RF', ylabel_1='Abweichung RF',
    ycolname_2='Abweichung RNN', ylabel_2='Abweichung RNN',
    col_name_vel='v_x', label_vel='$v$ in m/s', f_a=50, path='Plots'
):
    """
    Erstellt einen DIN 461 konformen Zeitverlaufsplan mit zwei Subplots:
    - Oben: Geschwindigkeit (25% der Höhe)
    - Unten: Stromverlauf mit Vorhersagen (75% der Höhe)
    """
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_orange = "#FFC000"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_magenta = "#A3107C"

    time = data.index / f_a

    # Erstelle eine Figure mit zwei Subplots (Höhenverhältnis 1:3)
    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.1)
    ax_vel = fig.add_subplot(gs[0])  # Geschwindigkeit (25%)
    ax_curr = fig.add_subplot(gs[1])  # Strom (75%)

    # --- Subplot für Geschwindigkeit ---
    ax_vel.spines['left'].set_position('zero')
    ax_vel.spines['bottom'].set_position('zero')
    ax_vel.spines['top'].set_visible(False)
    ax_vel.spines['right'].set_visible(False)
    ax_vel.spines['left'].set_color(kit_dark_blue)
    ax_vel.spines['bottom'].set_color(kit_dark_blue)
    ax_vel.spines['left'].set_linewidth(1.0)
    ax_vel.spines['bottom'].set_linewidth(1.0)
    ax_vel.tick_params(axis='x', colors=kit_dark_blue, direction='inout', length=6, labelbottom=False)
    ax_vel.tick_params(axis='y', colors=kit_dark_blue, direction='inout', length=6)
    ax_vel.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5, linestyle='-')
    ax_vel.set_axisbelow(True)

    line_vel, = ax_vel.plot(time, data[col_name_vel], label=label_vel, color=kit_magenta, linewidth=2)
    #ax_vel.set_ylabel(label_vel, color=kit_dark_blue, fontsize=12)
    ax_vel.set_ylim(bottom=data[col_name_vel].min() * 1.1, top=data[col_name_vel].max() * 1.1)

    # Achsenpfeile für den oberen Subplot (Geschwindigkeit)
    xmin_vel, xmax_vel = ax_vel.get_xlim()
    ymin_vel, ymax_vel = ax_vel.get_ylim()
    arrow_length_vel = 0.03 * (xmax_vel - xmin_vel)
    arrow_height_vel = 0.04 * (ymax_vel - ymin_vel)

    # X-Achse Pfeil (oben)
    ax_vel.annotate('', xy=(xmax_vel, -0.15 * ymax_vel), xytext=(xmax_vel * 0.95, -0.15 * ymax_vel),
                   arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5, mutation_scale=15))
    ax_vel.text(xmax_vel * 0.9, -0.2 * ymax_vel, r'$t$ in s',
                ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Y-Achse Pfeil (oben)
    ax_vel.annotate('', xy=(-0.04*xmax_vel, ymax_vel), xytext=(-0.04*xmax_vel, ymax_vel * 0.6),
                   arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5, mutation_scale=15))
    ax_vel.text(-0.05 * xmax_vel, ymax_vel * 0.2, label_vel,
                ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # --- Subplot für Strom und Vorhersagen ---
    ax_curr.spines['left'].set_position('zero')
    ax_curr.spines['bottom'].set_position('zero')
    ax_curr.spines['top'].set_visible(False)
    ax_curr.spines['right'].set_visible(False)
    ax_curr.spines['left'].set_color(kit_dark_blue)
    ax_curr.spines['bottom'].set_color(kit_dark_blue)
    ax_curr.spines['left'].set_linewidth(1.0)
    ax_curr.spines['bottom'].set_linewidth(1.0)
    ax_curr.tick_params(axis='x', colors=kit_dark_blue, direction='inout', length=6)
    ax_curr.tick_params(axis='y', colors=kit_dark_blue, direction='inout', length=6)
    ax_curr.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5, linestyle='-')
    ax_curr.set_axisbelow(True)

    # Plot der Hauptdaten (Strom)
    # Plot der Vorhersagen mit Standardabweichung
    def plot_prediction_with_std(ax, data, base_label, color, label=''):
        cols = [col for col in data.columns if col.startswith(base_label)]
        if not cols:
            return None, None
        mean = data[cols].mean(axis=1)
        std = data[cols].std(axis=1)
        line, = ax.plot(time, mean, label=label, color=color, linewidth=2)
        ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.2)
        return line, mean

    line1, mean1 = plot_prediction_with_std(ax_curr, data, ycolname_1, kit_red, ylabel_1)
    line2, mean2 = plot_prediction_with_std(ax_curr, data, ycolname_2, kit_orange, ylabel_2)

    line_curr, = ax_curr.plot(time, data[col_name], label=label, color=kit_blue, linewidth=2)

    # Achsenpfeile für den unteren Subplot (Strom)
    xmin_curr, xmax_curr = ax_curr.get_xlim()
    ymin_curr, ymax_curr = ax_curr.get_ylim()
    arrow_length_curr = 0.03 * (xmax_curr - xmin_curr)
    arrow_height_curr = 0.04 * (ymax_curr - ymin_curr)

    # X-Achse Pfeil (unten)
    ax_curr.annotate('', xy=(xmax_curr, -0.07 * ymax_curr), xytext=(xmax_curr * 0.95, -0.07 * ymax_curr),
                   arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5, mutation_scale=15))
    ax_curr.text(xmax_curr * 0.9, -0.1 * ymax_curr, r'$t$ in s',
                ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Y-Achse Pfeil (unten)
    ax_curr.annotate('', xy=(-0.04 * xmax_curr, ymax_curr), xytext=(-0.04 * xmax_curr, ymax_curr * 0.8),
                   arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5, mutation_scale=15))
    ax_curr.text(-0.05 * xmax_curr, ymax_curr * 0.7, axis_name,
                ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # Titel
    fig.suptitle(title, color=kit_dark_blue, fontsize=14, fontweight='bold', y=1.02)

    # Legende für den unteren Subplot
    lines_curr = [line for line in [line_curr, line1, line2] if line is not None]
    labels_curr = [line.get_label() for line in lines_curr]
    ax_curr.legend(lines_curr, labels_curr, loc='upper right', frameon=True, facecolor='white', edgecolor=kit_dark_blue)

    # Speichern des Plots
    plot_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved as {plot_path}')

# Beispielaufruf
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
    f'Aluminium {geometry}: Vorschubgeschwindigkeit und Stromverlauf der Vorschubachse in x-Richtung',
    f'Verlauf_{material}_{geometry}_Ref_RF_mit_Geschwindigkeit',
    col_name='curr_x', label='Messwerte', axis_name='$I$ in A',
    col_name_vel='v_x', label_vel='$v$ in m/s',
    ycolname_1='Random Forest', ylabel_1='Random Forest',
    ycolname_2='Rekurrentes neuronales Netz', ylabel_2='Rekurrentes neuronales Netz',
    dpi=1200
)
