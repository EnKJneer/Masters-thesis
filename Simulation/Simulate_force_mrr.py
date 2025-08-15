import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import monitoring
import machine_state
import matplotlib
matplotlib.use('TkAgg')

def split_curr_data(data: pd.DataFrame) -> tuple:
    curr_column = ['curr_x', 'curr_y', 'curr_z', 'curr_sp']
    true_curr = data[curr_column]
    process_value = data.drop(columns=curr_column)
    return true_curr, process_value

def get_part_properties(material_geometry):
    # Initialize part properties
    part_position = None
    part_dimension = None

    material_geometry = material_geometry.replace('.csv', '')
    # Split the material_geometry string into components
    material, *geometry = material_geometry.split('_')

    # Determine part properties based on material and geometry
    if material == "S235JR":
        material = "S235JR"
        if "Gear" in geometry:
            if "Depth" in geometry:
                part_position = [-33.15, 174.482, 354.1599] #[-40, 174.871, 354.78]
            else:
                part_position = [-33.807, 174.871, 354.78]
            part_dimension = [70, 70, 50, 0.1]
        elif "Notch" in geometry:
            part_position = [-38.6, 249.5, 354.93]
            part_dimension = [70, 70, 50, 0.1]
        elif "Plate" in geometry:
            part_position = [-38.64, 175, 354.94]
            part_dimension = [75.0, 75.0 * 2, 49.6, 0.1]
        elif "Kühlgrill" in geometry:
            part_position = [-39.243, 176.407, 351.205]
            part_dimension = [75.0, 75.0 * 2, 50.0, 0.1]
        elif "Laufrad" in geometry:
            part_position = [-76.69, 175.4, 360]
            part_dimension = [150, 150, 30, 0.1]
        elif "Bauteil_1" in geometry:
            part_position = [-37.8, 269.27, 339.69]
            part_dimension = [70, 70, 50, 0.1]

    elif material == "AL" and geometry[:2] == ["2007", "T4"]:
        material = "AL_2007_T4"
        if "Gear" in geometry:
            if "Depth" in geometry:
                part_position = [-33.24875, 174.482, 355.1599]
            else:
                part_position = [-33.807, 174.871, 354.78]
            part_dimension = [70, 70, 50, 0.1]
        elif "Notch" in geometry:
            part_position = [-38.6, 249.5, 354.93]
            part_dimension = [70, 70, 50, 0.1]
        elif "Plate" in geometry:
            part_position = [-35.64, 175, 354.94]
            part_dimension = [75.0, 75.0 * 2, 50.0, 0.1]
        elif "Kühlgrill" in geometry:
            part_position = [-39.243, 176.407, 351.205]
            part_dimension = [75.0, 75.0 * 2, 50.0, 0.1]
        elif "Laufrad" in geometry:
            part_position = [-76.69, 175.4, 360]
            part_dimension = [150, 150, 30, 0.1]
        elif "Bauteil_1" in geometry:
            part_position = [-37.8, 269.27, 339.69]
            part_dimension = [70, 70, 50, 0.1]

    if part_position is not None and part_dimension is not None and material is not None:
        return material,part_position, part_dimension
    else:
        print(f"Unknown material and geometry combination: {material_geometry}")
        return "Unknown material and geometry combination"

def plot_time_series(data, title, dpi=300, col_name='v_x', label='Messwerte',
                     label_axis='Geschwindigkeit in m/s',
                     col_name1=None, label1='Simulation', col_name2=None, label2='Simulation 2',
                     f_a=50):
    """
    Erstellt einen DIN 461 konformen Zeitverlaufsplan.
    :param data: DataFrame mit den Daten
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
    x_label_pos = xmax# * 0.95
    y_label_pos = -0.08 * (ymax - ymin)

    ax1.annotate('', xy=(x_label_pos + arrow_length, y_label_pos),
                 xytext=(x_label_pos , y_label_pos),
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

    # DIN 461: Achsenbegrenzungen anpassen, damit Nullpunkt sichtbar ist
    ax1.set_xlim(left=min(x_label_pos_y, xmin), right=xmax * 1.05)
    ax1.set_ylim(bottom=min(y_label_pos, ymin), top=ymax * 1.05)

    plt.show()

if __name__ == '__main__':
    tool_diameter = 10
    target_frequency = 50
    path_material_constant = 'optimized_parameters.json'

    show_results = True
    path = '..\\DataSets\\DataMerged'
    path_target = '..\\DataSets\\DataSimulated_test'
    # Create target directory if it doesn't exist
    os.makedirs(path_target, exist_ok=True)
    files = os.listdir(path)
    files = ['S235JR_Gear_Depth.csv']
    for file in files:
        if not file.endswith('.csv'):
            continue

        print(f'Processing {file}')

        path_data = os.path.join(path, file)
        raw_data = pd.read_csv(path_data)

        material, part_position, part_dimension = get_part_properties(file.replace('.csv', ''))

        material_setting = machine_state.load_optimized_parameters_as_dict(path_material_constant)

        new_machine_state, tool, part = machine_state.set_machine_state(material_setting, material,
                                                                        tool_diameter, part_position,
                                                                        part_dimension)
        true_curr, process_value = split_curr_data(raw_data)

        data_df = monitoring.state_monitoring(new_machine_state, tool,
                                              part, process_value, true_curr,
                                              target_frequency, part_position, part_dimension, show_results)
        path_data = os.path.join(path_target, file)
        data_df.to_csv(path_data)
        if show_results:
            data_df['f_x'] = -data_df['f_x']*200
            plot_time_series(data_df, f'Verlauf der Prozesskraft',
                             col_name='f_x', label='Messwerte', label_axis='$F$ in N', dpi=300,
                             col_name1='f_x_sim', label1='Simulation')
            plot_time_series(data_df, f'Verlauf der Prozesskraft',
                             col_name='f_x', label='Messwerte', label_axis='$F$ in N', dpi=300,
                             col_name1='materialremoved_sim', label1='materialremoved')