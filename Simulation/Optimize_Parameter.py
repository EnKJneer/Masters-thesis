import glob
import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.f2py.auxfuncs import throw_error
from scipy.optimize import minimize

def calculate_theta(v_x, v_y):
    if v_x == 0 and v_y == 0:
        return 0
    theta = math.atan2(v_y, v_x)
    if theta < 0:
        theta += 2 * math.pi
    return theta

def calculate_force(input, x, y, z, k_c1, k_f1, k_p1, K, machine_coef_x, machine_coef_y, machine_coef_z) -> tuple:
    epsilon = 0.25
    # Konstanten
    digits = 3
    tool_radius = 5 # tool
    a_e = 2 * tool_radius # tool -> Länge des tools der Fräst bei uns der Durchmesser
    tooth_amount = 4 # tool
    kappa = math.radians(90)  # kappe = 90 da Umfangfräsen
    phi_s = math.radians(180)  # Schnittwinkel beim volle fraesen ist das 180 Grad
    #K = 0.9
    #k_c1 = 1780
    #k_f1 = 351
    #k_p1 = 274
    #x = 0.7019
    #y = 0.4911
    #z = 0.26

    # Input
    v_x, v_y, v_z, v_sp, a_p = input
    v_ges = math.sqrt(v_x**2 + v_y**2 + v_z**2)
    v_sp = v_sp * 10 / (60 * 60)

    if abs(v_sp) < epsilon or math.sqrt(v_x**2 + v_y**2) < epsilon:
        return 0, 0, 0

    theta = calculate_theta(v_x, v_y)

    force_rotation = np.array([[math.cos(theta), math.sin(theta), 0],
                               [math.sin(theta), -math.cos(theta), 0],
                               [0, 0, 1]])

    fz = v_ges / (v_sp * tooth_amount)  # Vorschub pro Zahn

    h = 114.6 / math.degrees(phi_s) * fz * (a_e / (tool_radius * 2)) * math.sin(kappa)  # Formel 3.11 nach Neugebauer seite 41
    b = a_p / math.sin(kappa)
    h = max(h, 0)

    F_c = b * h ** (1 - z) * k_c1 * K
    F_cn = b * h ** (1 - x) * k_f1 * K
    F_pz = b * h ** (1 - y) * k_p1 * K

    F_c_matrix = np.array([[F_c], [F_cn], [F_pz]])

    f_xyz = force_rotation @ F_c_matrix
    f_x = f_xyz[0, 0] * machine_coef_x
    f_y = f_xyz[1, 0] * machine_coef_y
    f_z = f_xyz[2, 0] * machine_coef_z

    f_sp = math.sqrt(f_x ** 2 + f_y ** 2)

    return round(f_x, digits), round(f_y, digits), round(f_z, digits), round(f_sp, digits)

def objective_function(params, x, y):
    forces = [calculate_force(row, *params) for _, row in x.iterrows()]

    predicted_forces = np.array(forces)

    error = np.sum(100*(predicted_forces[:, 0] - y['f_x'].values) ** 2
                   + 100*(predicted_forces[:, 1] - y['f_y'].values) ** 2
                   + (predicted_forces[:, 2] - y['f_z'].values) ** 2
                   + 100*(predicted_forces[:, 3] - y['f_sp'].values) ** 2)
    print(error)
    return error

def filter_data(raw_data, part_dimension):
    mean_v_sp = raw_data['v_sp'].mean()
    std_v_sp = raw_data['v_sp'].std()

    # Festlegung der Grenzen für die Filterung
    lower_bound = mean_v_sp - 4 * std_v_sp
    upper_bound = mean_v_sp + 4 * std_v_sp

    # Filtern der Daten
    filtered_data = raw_data[(raw_data['v_sp'] >= lower_bound) & (raw_data['v_sp'] <= upper_bound)].reset_index(drop=True)
    filtered_data = filtered_data[filtered_data['v_sp'] > 1].reset_index(drop=True)
    filtered_data = filtered_data[abs(filtered_data['v_z']) < 0.25].reset_index(drop=True)
    filtered_data = filtered_data[abs(filtered_data['f_x']) > 0.8].reset_index(drop=True)
    # Berechnung der halben Abmessungen
    half_dim_x = part_dimension[0]
    half_dim_y = part_dimension[1]
    half_dim_z = part_dimension[2]

    # Überprüfung, ob die Position innerhalb der part_dimensions liegt
    def is_within_dimensions(pos_x, pos_y, pos_z, half_dim_x, half_dim_y, half_dim_z):
        return (abs(pos_x) <= half_dim_x) and (abs(pos_y) <= half_dim_y) and (abs(pos_z) <= half_dim_z)

    # Anwendung des Filters auf die Daten
    filtered_data = filtered_data[filtered_data.apply(
        lambda row: is_within_dimensions(
            row['pos_x'] - part_position[0],
            row['pos_y'] - part_position[1],
            row['pos_z'] - part_position[2],
            half_dim_x, half_dim_y, half_dim_z
        ), axis=1
    )].reset_index(drop=True)

    return filtered_data

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

def seperate_data(input_data):
    data = input_data.copy()
    data['f_sp'] = (data['f_x'] ** 2 + data['f_y'] ** 2)**0.5
    y = data[['f_x', 'f_y', 'f_z', 'f_sp']].copy() *200 # Alte Kräfte sind um 200 Skaliert -> Übereinstimmung mit alten Parametern
    y.loc[:, ['f_x', 'f_z']] *= -1
    x = data[['v_x', 'v_y', 'v_z', 'v_sp', 'a_p']].copy()
    return x, y

def get_initial_params(material):
    # lade Parameter als initalwerte
    path_material_constant = 'material_constant.csv'
    if material == 'S235JR':
        material_choose = 0
    elif material == 'AL_2007_T4':
        material_choose = 1
    else:
        material_choose = int(input("Material wählen:"))

    material_setting = pd.read_csv(path_material_constant, sep=';')

    # Initalparameter
    k_c1 = material_setting.loc[material_choose, 'k_c1']
    k_f1 = material_setting.loc[material_choose, 'k_f1']
    k_p1 = material_setting.loc[material_choose, 'k_p1']
    x_param = material_setting.loc[material_choose, 'x']
    y_param = material_setting.loc[material_choose, 'y']
    z_param = material_setting.loc[material_choose, 'z']
    # selbstgewählte Parameter funktionieren besser
    machine_coef_x = material_setting.loc[material_choose, 'machine_coef_x']
    machine_coef_y = material_setting.loc[material_choose, 'machine_coef_y']
    machine_coef_z = material_setting.loc[material_choose, 'machine_coef_z']

    return  [x_param, y_param, z_param, k_c1, k_f1, k_p1, 0.9, machine_coef_x, machine_coef_y, machine_coef_z]

def get_part_properties(material_geometry):
    # Initialize part properties
    part_position = None
    part_dimension = None
    material = None

    # Split the material_geometry string into components
    material, *geometry = material_geometry.split('_')

    # Determine part properties based on material and geometry
    if material == "S235JR":
        material = "Stahl"
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

def plot_forces(forces, y, index, label, ref_label=None, ref_forces=None):
    plt.figure(figsize=(12, 8))
    plt.plot(forces[:, index], label=f'Force {label}')
    if ref_forces is not None:
        plt.plot(ref_forces[:, index], label=f'Force {label} - ref')
    plt.plot(y[f'f_{label}'], label=f'gt {label}', linestyle='--', alpha=0.8)
    plt.legend()
    plt.title(f'Calculated Forces in {label.upper()} Direction')
    plt.xlabel('Sample')
    plt.ylabel('Force')
    plt.show()

def save_named_optimized_parameters(material: str, params: list, filename='optimized_parameters.json'):
    """
    Speichert benannte optimierte Parameter für ein Material als JSON-Datei.

    :param material: Name des Materials
    :param params: Liste der optimierten Parameter (Reihenfolge muss stimmen)
    :param filename: Pfad zur JSON-Datei
    """
    param_keys = [
        "x", "y", "z",
        "k_c1", "k_f1", "k_p1",
        "K",
        "machine_coef_x", "machine_coef_y", "machine_coef_z",
        "t_x", "t_y", "t_z"
    ]

    if len(params) != len(param_keys):
        raise ValueError(f"Erwarte {len(param_keys)} Parameter, aber {len(params)} erhalten.")

    param_dict = dict(zip(param_keys, params))

    # Bestehende Datei laden oder neues Dict beginnen
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[material] = param_dict

    # In Datei schreiben
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Optimierte Parameter für '{material}' erfolgreich in '{filename}' gespeichert.")

if __name__ == '__main__':

    part_position = [-33.15, 174.482, 354.1599] # gear # [-35.64, 175.0, 354.94] Plate
    part_dimension = [70, 70, 50, 0.2] # gear # [75.0, 75.0*2, 50.0, 0.1] Plate

    material = 'S235JR'

    print('load data')
    path = '..\\DataSetsV3\\DataMerged'

    # Alle Dateien, die mit "S235JR" beginnen und ".csv" enden
    all_files = glob.glob(os.path.join(path, f'{material}*.csv'))

    # Testdateien ausschließen
    exclude_files = {f'{material}_Notch_Normal.csv'}

    # Nur die gewünschten Trainingsdateien
    files = [os.path.basename(f) for f in all_files if os.path.basename(f) not in exclude_files]

    x_all = []
    y_all = []

    for file in files:
        path_data = os.path.join(path, file)
        raw_data = pd.read_csv(path_data)
        if file.startswith('AL_') and file.endswith('Depth.csv'):
            print('Schneide Fehlerhafte bereiche raus')
            n = int(2/3 *len(raw_data))
            raw_data = raw_data.iloc[:n].reset_index(drop=True)

        _, part_position, part_dimension = get_part_properties(file)
        filtered_data = filter_data(raw_data, part_dimension)

        filtered_data['a_p'] = calculate_a_p(filtered_data['pos_z'], part_position[2], part_dimension[2])

        x, y = seperate_data(filtered_data)
        x_all.append(x.reset_index(drop=True))
        y_all.append(y.reset_index(drop=True))

    x = pd.concat(x_all, ignore_index=True)
    y = pd.concat(y_all, ignore_index=True)

    plt.figure(figsize=(12, 8))
    plt.plot(y['f_x'], label='f_x')
    plt.legend()
    plt.show()

    initial_params = get_initial_params(material)

    # 1. Optimiere Parameter
    # Optionen für die Optimierung
    options = {
        'maxiter': 50000,  # Maximale Anzahl von Iterationen
        'fatol': 1e-3,  # Toleranz für Änderungen in der Zielfunktion
        'xatol': 1e-3,  # Toleranz für Änderungen in den Parametern
        'disp': True  # Fortschrittsinformationen anzeigen
    }

    result = minimize(objective_function, initial_params, args=(x, y), method='Nelder-Mead', options=options) #SLSQP Nelder-Mead dogleg L-BFGS-B
    optimized_params = result.x
    print(optimized_params)
    save_named_optimized_parameters(material, optimized_params.tolist())

    # 2.1 Berechne Kräfte mit optimierten Parametern
    forces = [calculate_force(row, *optimized_params) for _, row in x.iterrows()]
    forces = np.array(forces)
    '''
    # 2.2 Berechne Referenz Kräfte mit initialen Parametern
    forces_ref = [calculate_force(row, *initial_params) for _, row in x.iterrows()]
    forces_ref = np.array(forces_ref)'''

    # 3. Plotte Kräfte
    for i, axis in enumerate(['x', 'y', 'z']):
        plot_forces(forces, y, i, axis, ref_forces=None)

    # test
    print('Starte Test')
    file = f'{material}_Notch_Depth.csv'
    path_data = os.path.join(path, file)
    raw_data = pd.read_csv(path_data)

    _, part_position, part_dimension = get_part_properties(file)
    #filtered_data = filter_data(raw_data, part_dimension).reset_index(drop=True)

    raw_data['a_p'] = calculate_a_p(raw_data['pos_z'], part_position[2], part_dimension[2])

    plt.figure(figsize=(12, 8))
    plt.plot(raw_data['a_p'], label='a_p')
    plt.legend()
    plt.show()

    x, y = seperate_data(raw_data)

    forces = [calculate_force(row, *optimized_params) for _, row in x.iterrows()]
    forces = np.array(forces)

    for i, axis in enumerate(['x', 'y', 'z']):
        plot_forces(forces, y, i, axis, ref_forces=None)