import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import machine_state as ms

from MMR_Calculator import voxel_class_numba as vc
from Simulation.MMR_Calculator.MRRSimulationV3 import SimpleCNCMRRSimulation

'''def check_and_correct_values(vector):
    valid_values = {0, 3, 6, 12}
    corrected_vector = []

    for value in vector:
        if value in valid_values:
            corrected_vector.append(value)
        else:
            # Finde den nächstgelegenen gültigen Wert
            closest_value = min(valid_values, key=lambda x: abs(x - value))
            corrected_vector.append(closest_value)
            print(f"Warnung: Ungültiger Wert {value} gefunden und auf {closest_value} korrigiert.")

    return np.array(corrected_vector)

def calculate_a_p(pos_z, z_position, z_dimension):
    # Berechnung von output
    output = -np.round(pos_z - z_position - z_dimension, 1)

    # Setze Werte auf 0, bei denen abs(pos_z - z_position) nicht kleiner als z_dimension ist
    output = np.where(np.abs(pos_z - z_position) < z_dimension, output, 0)

    # Zähle die Häufigkeit jedes eindeutigen Werts
    unique_pos_z, counts = np.unique(output, return_counts=True)

    # Filtere Werte, von denen es mindestens 20 Stück gibt
    frequent_unique_pos_z = unique_pos_z[counts >= 20]

    # Sortiere die gefilterten Werte
    sorted_unique_pos_z = np.sort(frequent_unique_pos_z)

    # Berechne die neuen Werte
    new_pos_z_values = np.zeros_like(sorted_unique_pos_z, dtype=float)
    for i in range(len(sorted_unique_pos_z)):
        if i == 0:
            new_pos_z_values[i] = sorted_unique_pos_z[i]
        else:
            new_pos_z_values[i] = sorted_unique_pos_z[i] - sorted_unique_pos_z[i - 1]
    new_pos_z_values = check_and_correct_values(new_pos_z_values)

    print("Korrigierter Vektor:", new_pos_z_values)
    # Erstelle ein Mapping von alten zu neuen Werten
    value_mapping = {old: new for old, new in zip(sorted_unique_pos_z, new_pos_z_values)}

    # Ersetze die Werte in 'output' durch die neuen Werte
    for i in range(len(output)):
        if output[i] in value_mapping:
            output[i] = value_mapping[output[i]]

    return output
'''

def calculate_ae_and_angle_for_tool(tool, part_position, part_dimension):
    """
    Berechnet a_e (Schnittlänge) und Schnittwinkel für das gegebene Tool und Bauteil.

    tool: Tool-Objekt mit radius, x_position, y_position
    part_position: [x_min, y_min, z_min] untere linke Ecke des Bauteils
    part_dimension: [width, height, depth] Bauteilmaße
    """
    # Werte extrahieren
    tool_x = tool.x_position
    tool_y = tool.y_position
    radius = tool.radius

    x_min = part_position[0]
    y_min = part_position[1]
    x_max = x_min + part_dimension[0]
    y_max = y_min + part_dimension[1]

    # Vollständig draußen?
    if (tool_x + radius <= x_min or tool_x - radius >= x_max or
        tool_y + radius <= y_min or tool_y - radius >= y_max):
        return 0.0, 0.0

    # Vollständig drin?
    if (tool_x - radius >= x_min and tool_x + radius <= x_max and
        tool_y - radius >= y_min and tool_y + radius <= y_max):
        return 2 * radius, 180.0

    # Teilweise drin -> Geometrische Berechnung
    max_chord = 0.0
    step = radius / 180.0  # Scan-Auflösung in Y
    for y in frange(max(y_min, tool_y - radius), min(y_max, tool_y + radius), step):
        dy = y - tool_y
        dx = math.sqrt(max(radius**2 - dy**2, 0))
        x_left = tool_x - dx
        x_right = tool_x + dx
        chord_left = max(x_left, x_min)
        chord_right = min(x_right, x_max)
        chord_length = max(chord_right - chord_left, 0)
        max_chord = max(max_chord, chord_length)

    a_e = max_chord

    # Schnittwinkel berechnen (aus Geradenlänge in Kreis)
    if a_e > 0:
        angle = math.degrees(2 * math.asin(min(a_e / (2 * radius), 1.0)))
    else:
        angle = 0.0

    return a_e, angle

def frange(start, stop, step):
    """Float range generator."""
    while start <= stop:
        yield start
        start += step

def check_and_correct_values(vector):
    valid_values = {0, 3, 6, 12}
    corrected_vector = []

    for value in vector:
        if value in valid_values:
            corrected_vector.append(value)
        else:
            # Finde den nächstgelegenen gültigen Wert
            closest_value = min(valid_values, key=lambda x: abs(x - value))
            corrected_vector.append(closest_value)
            print(f"Warnung: Ungültiger Wert {value} gefunden und auf {closest_value} korrigiert.")

    return np.array(corrected_vector)

def check_and_correct_values_2(vector):
    valid_values = {0, 3, 6, 12}
    corrected_vector = []
    previous_value = None
    for value in vector:
        if value in valid_values:
            corrected_vector.append(value)
        else:
            # Ersetze den ungültigen Wert durch den vorherigen Wert im Array
            if previous_value is not None:
                corrected_vector.append(previous_value)
                print(f"Warnung: Ungültiger Wert {value} gefunden und auf {previous_value} korrigiert.")
            else:
                # Falls es keinen vorherigen Wert gibt, verwende den nächstgelegenen gültigen Wert
                closest_value = min(valid_values, key=lambda x: abs(x - value))
                corrected_vector.append(closest_value)
                print(f"Warnung: Ungültiger Wert {value} gefunden und auf {closest_value} korrigiert.")
        previous_value = corrected_vector[-1]
    return np.array(corrected_vector)

def calculate_a_p(pos_z, z_position, z_dimension):
    # Berechnung von output
    output = -np.round(pos_z - z_position - z_dimension, 1)

    # Setze Werte auf 0, bei denen abs(pos_z - z_position) nicht kleiner als z_dimension ist
    output = np.where(np.abs(pos_z - z_position) < z_dimension, output, 0)

    # Zähle die Häufigkeit jedes eindeutigen Werts
    unique_pos_z, counts = np.unique(output, return_counts=True)

    # Filtere Werte, von denen es mindestens 20 Stück gibt
    frequent_unique_pos_z = unique_pos_z[counts >= 20]

    # Sortiere die gefilterten Werte
    sorted_unique_pos_z = np.sort(frequent_unique_pos_z)

    # Berechne die neuen Werte
    new_pos_z_values = np.zeros_like(sorted_unique_pos_z, dtype=float)
    for i in range(len(sorted_unique_pos_z)):
        if i == 0:
            new_pos_z_values[i] = sorted_unique_pos_z[i]
        else:
            new_pos_z_values[i] = sorted_unique_pos_z[i] - sorted_unique_pos_z[i - 1]

    print("Vektor:", new_pos_z_values)

    new_pos_z_values = check_and_correct_values(new_pos_z_values)

    print("Korrigierter Vektor:", new_pos_z_values)

    # Erstelle ein Mapping von alten zu neuen Werten
    value_mapping = {old: new for old, new in zip(sorted_unique_pos_z, new_pos_z_values)}

    # Ersetze die Werte in 'output' durch die neuen Werte
    for i in range(len(output)):
        if output[i] in value_mapping:
            output[i] = value_mapping[output[i]]

    # Überprüfe den gesamten Array am Ende
    final_output = check_and_correct_values_2(output)
    print("Finaler korrigierter Vektor:", final_output)

    return final_output

def calculate_average(parameter_list: list, parameter_new, amount: int = 50, mrr: bool = False) -> float:
    parameter_list.append(parameter_new)
    if len(parameter_list) > amount:
        parameter_list.pop(0)
    if mrr:
        if parameter_new == 0 or parameter_new > 50:
            parameter_average = 0
        else:
            parameter_average = sum(parameter_list)/len(parameter_list)
    else:
        parameter_average = sum(parameter_list)/len(parameter_list)
    return parameter_average

def is_tool_in_part(tool_radius, a_p, pos_x, pos_y, pos_z, part_position, part_dimension):
    """Prüft, ob das Werkzeug vollständig im Werkstück liegt (X/Y mit Radius, Z mit Schnitttiefe)."""
    x_min = pos_x - tool_radius
    x_max = pos_x + tool_radius
    y_min = pos_y - tool_radius
    y_max = pos_y + tool_radius
    z_min = pos_z - a_p /10
    z_max = pos_z + a_p /10  # Werkzeugspitze

    part_x_min = part_position[0]
    part_x_max = part_position[0] + part_dimension[0]
    part_y_min = part_position[1]
    part_y_max = part_position[1] + part_dimension[1]
    part_z_min = part_position[2]
    part_z_max = part_position[2] + part_dimension[2]

    return (
            x_min >= part_x_min and x_max <= part_x_max and
            y_min >= part_y_min and y_max <= part_y_max and
            z_min >= part_z_min and z_max <= part_z_max
    )
def state_monitoring(machine_state: ms.MachineState, tool: vc.Tool, part: vc.PartialVoxelPart,
                     process_data: pd.DataFrame, true_curr: pd.DataFrame, frequence: int, part_position, part_dimension, plot_mrr = False) -> pd.DataFrame:

    a_e = machine_state.get_tool_radius() * 2
    a_p_array = calculate_a_p(process_data['pos_z'], part_position[2], part_dimension[2])
    process_data['a_p'] = a_p_array
    force_df = pd.DataFrame()
    new_part_volume =part_dimension[0] * part_dimension[1] * part_dimension[2]
    print(f'Volumen vorher: {new_part_volume}')

    #simulator = SimpleCNCMRRSimulation(part_position, part_dimension, tool.radius)
    #times, mrr_values, segments  = simulator.simulate_mrr(process_data, frequence)
    #if plot_mrr:
    #    simulator.plot_results(times, mrr_values, segments, process_data)
    simulator = SimpleCNCMRRSimulation(part_position, part_dimension, tool.radius, frequence)
    times, mrr_values  = simulator.simulate_mrr(process_data)
    if plot_mrr:
        simulator.plot_results(times, mrr_values)

    mrr_mean = mrr_values.mean()
    for i in range(process_data.shape[0]):
        a_p = a_p_array[i]
        pos_x = process_data.loc[i, 'pos_x']
        pos_y = process_data.loc[i, 'pos_y']
        pos_z = process_data.loc[i, 'pos_z']
        new_tool_coordinates = [pos_x, pos_y, pos_z]
        v_x = process_data.loc[i, 'v_x']
        v_y = process_data.loc[i, 'v_y']
        v_z = process_data.loc[i, 'v_z']
        v_sp = process_data.loc[i, 'v_sp']

        a_e, phi = calculate_ae_and_angle_for_tool(tool, part_position, part_dimension)

        process_data.loc[i, 'a_e'] = a_e
        process_data.loc[i, 'cut_angle'] = phi

        current_process_state = ms.ProcessState(pos_x, pos_y, pos_z, v_x, v_y, v_z, v_sp, a_p, a_e, phi)
        tool.set_new_position(new_tool_coordinates)
        is_in = is_tool_in_part(tool.radius, a_p, pos_x, pos_y, pos_z, part_position, part_dimension)
        materialremoved_sim = mrr_values[i]
        if materialremoved_sim < 0:
            materialremoved_sim = 0

        if materialremoved_sim > mrr_mean/10 and a_p > 0 and a_e > 2:
            f_x_sim, f_y_sim, f_z_sim, f_sp_sim = force_calculate(machine_state, current_process_state, frequence)
        else:
            f_x_sim, f_y_sim, f_z_sim, f_sp_sim = 0.0, 0.0, 0.0, 0.0

        status = round(i / process_data.shape[0] * 100, 2)
        print(f'Satus: {status} f_x_sim: {f_x_sim}')
        force_df = pd.concat([force_df, test_update_force_df(f_x_sim, f_y_sim, f_z_sim, f_sp_sim, materialremoved_sim)], ignore_index=True)

    if plot_mrr:
        plt.plot(process_data['a_p'])
        plt.title('a_p')
        plt.show()

    data_df = pd.concat([process_data, true_curr, force_df], axis=1)
    data_df = data_df.rename(columns={'pos_x': 'pos_x', 'pos_y': 'pos_y', 'pos_z': 'pos_z', 'MRR': 'materialremoved_sim',
                                      'f_x_sim': 'f_x_sim', 'f_y_sim': 'f_y_sim', 'f_z_sim': 'f_z_sim', 'f_sp_sim': 'f_sp_sim'})

    cols = ['f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim'] #, 'materialremoved_sim'
    alphas = [0.5, 0.5, 0.5, 0.5, 0.9 ]
    for idx, col in enumerate(cols):
        data_df[col] = exponential_smoothing(data_df[col], alphas[idx])

    removed_material = mrr_values.cumsum() * 1/simulator.sampling_frequency
    print(f'Entferntes Material: {removed_material[-1]}')
    print(f'Volumen nach der Bearbeitung: {new_part_volume - removed_material[-1]}')
    print(f'Prozent entfernt: {round(removed_material[-1] / new_part_volume *100, 2)} %')
    return data_df

def exponential_smoothing(data: np.ndarray, alpha: float) -> np.ndarray:
    """Apply exponential smoothing to the data."""
    smoothed_data = np.zeros_like(data)
    if len(data) > 0:
        smoothed_data[0] = data[0]  # Initial value
        for i in range(1, len(data)-1):
            smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1] + (1 - alpha) * smoothed_data[i + 1]
    return smoothed_data

def force_calculate(machine_state: ms.MachineState, process_state: ms.ProcessState, frequence: int) -> tuple:
    return process_state.calculate_force(machine_state, frequence)

def test_update_force_df(f_x, f_y, f_z, f_sp, mrr) -> pd.DataFrame:
    temp_force_df = pd.DataFrame([[f_x, f_y, f_z, f_sp, mrr]], columns=['f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim', 'MRR'])
    return temp_force_df