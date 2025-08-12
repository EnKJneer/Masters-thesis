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

    # Split the material_geometry string into components
    material, *geometry = material_geometry.split('_')

    # Determine part properties based on material and geometry
    if material == "S235JR":
        material = "S235JR"
        if "Gear" in geometry:
            if "Depth" in geometry:
                part_position = [-33.15, 174.482, 354.1599]
            else:
                part_position = [-33.807, 174.871, 354.78]
            part_dimension = [70, 70, 50, 0.2]
        elif "Notch" in geometry:
            part_position = [-38.6, 249.5, 354.93]
            part_dimension = [70, 70, 50, 0.2]
        elif "Plate" in geometry:
            part_position = [-38.64, 175, 354.94]
            part_dimension = [75.0, 75.0 * 2, 49.6, 0.2]
        elif "Kühlgrill" in geometry:
            part_position = [-39.243, 176.407, 351.205]
            part_dimension = [75.0, 75.0 * 2, 50.0, 0.2]
        elif "Laufrad" in geometry:
            part_position = [-76.69, 175.4, 360]
            part_dimension = [150, 150, 30, 0.2]
        elif "Bauteil_1" in geometry:
            part_position = [-37.8, 269.27, 339.69]
            part_dimension = [70, 70, 50, 0.2]

    elif material == "AL" and geometry[:2] == ["2007", "T4"]:
        material = "AL_2007_T4"
        if "Gear" in geometry:
            if "Depth" in geometry:
                part_position = [-33.24875, 174.482, 355.1599]
            else:
                part_position = [-33.807, 174.871, 354.78]
            part_dimension = [70, 70, 50, 0.2]
        elif "Notch" in geometry:
            part_position = [-38.6, 249.5, 354.93]
            part_dimension = [70, 70, 50, 0.2]
        elif "Plate" in geometry:
            part_position = [-35.64, 175, 354.94]
            part_dimension = [75.0, 75.0 * 2, 50.0, 0.2]
        elif "Kühlgrill" in geometry:
            part_position = [-39.243, 176.407, 351.205]
            part_dimension = [75.0, 75.0 * 2, 50.0, 0.2]
        elif "Laufrad" in geometry:
            part_position = [-76.69, 175.4, 360]
            part_dimension = [150, 150, 30, 0.2]
        elif "Bauteil_1" in geometry:
            part_position = [-37.8, 269.27, 339.69]
            part_dimension = [70, 70, 50, 0.2]

    if part_position is not None and part_dimension is not None and material is not None:
        return material,part_position, part_dimension
    else:
        print(f"Unknown material and geometry combination: {material_geometry}")
        return "Unknown material and geometry combination"
    
def plot_time_series(data, title, dpi=300, label='v_x', ylabel='curr_x', f_a=50, align_axis=False):
    """
    Erstellt einen Zeitverlaufsplan mit zwei y-Achsen.

    :param data: DataFrame mit den Daten
    :param title: Titel des Plots
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    :param label: Bezeichnung der ersten y-Achse
    :param ylabel: Bezeichnung der zweiten y-Achse
    :param f_a: Abtastrate
    """
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=dpi)

    if 'time' in data.columns:
        time = data['time']
    else:
        time = data.index * (1/f_a)

    ax1.plot(time, data[label], label=label, color='tab:green')
    ax1.set_xlabel('Time in s')
    ax1.set_ylabel(label)
    ax1.set_title(title)
    ax1.legend(loc='upper left')

    if ylabel is not None:
        # Zweite y-Achse für curr_x
        ax2 = ax1.twinx()
        ax2.plot(time, data[ylabel], label=ylabel, color='tab:red', linestyle='--', alpha=0.8)
        ax2.set_ylabel(ylabel)
        ax2.legend(loc='upper right')

        if align_axis:
            # Berechne den maximalen absoluten Wert für jede Achse separat
            y1_min, y1_max = ax1.get_ylim()
            y2_min, y2_max = ax2.get_ylim()

            abs_max1 = max(abs(y1_min), abs(y1_max))
            abs_max2 = max(abs(y2_min), abs(y2_max))

            ax1.set_ylim(-abs_max1, abs_max1)
            ax2.set_ylim(-abs_max2, abs_max2)
            lim = 1000
            ax1.set_ylim(-lim, lim)
            ax2.set_ylim(-lim, lim)

    plt.show()

if __name__ == '__main__':
    tool_diameter = 10
    target_frequency = 50
    path_material_constant = 'optimized_parameters.json'

    show_results = False
    path = '..\\DataSets\\DataMerged'
    path_target = '..\\DataSets\\DataSimulated_low_res'

    # Create target directory if it doesn't exist
    os.makedirs(path_target, exist_ok=True)
    files = os.listdir(path)
    #files = ['AL_2007_T4_Gear_Normal.csv']

    for file in files:
        if not file.endswith('.csv'):
            continue

        print(f'Processing {file}')

        path_data = os.path.join(path, file)
        raw_data = pd.read_csv(path_data)

        material, part_position, part_dimension = get_part_properties(file)

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
            plot_time_series(data_df, file, label='f_x_sim', dpi=300, ylabel='f_x', align_axis=True)
            plot_time_series(data_df, file, label='f_x_sim', dpi=300, ylabel='materialremoved_sim', align_axis=False)