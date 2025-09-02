import glob
import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.f2py.auxfuncs import throw_error
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

def is_tool_in_part(data, tool_radius, part_position, part_dimension):
    """
    Gibt eine Maske zurück, die angibt, ob das Werkzeug vollständig im Werkstück liegt.
    """
    tool_radius = tool_radius *2 # sciherheitsfaktor
    pos_x = data['pos_x']
    pos_y = data['pos_y']
    pos_z = data['pos_z']
    a_p = data['a_p']

    x_min = pos_x - tool_radius
    x_max = pos_x + tool_radius
    y_min = pos_y - tool_radius
    y_max = pos_y + tool_radius
    z_min = pos_z - a_p / 10  # Annahme: a_p ist die Schnitttiefe
    z_max = pos_z  # Werkzeugspitze liegt bei pos_z (da a_p nach unten geht)

    part_x_min = part_position[0]
    part_x_max = part_position[0] + part_dimension[0]
    part_y_min = part_position[1]
    part_y_max = part_position[1] + part_dimension[1]
    part_z_min = part_position[2]
    part_z_max = part_position[2] + part_dimension[2]

    mask = (
        (x_min >= part_x_min) &
        (x_max <= part_x_max) &
        (y_min >= part_y_min) &
        (y_max <= part_y_max) &
        (z_min >= part_z_min) &
        (z_max <= part_z_max)
    )
    return mask

def calculate_theta(v_x, v_y):
    if v_x == 0 and v_y == 0:
        return 0
    theta = math.atan2(v_y, v_x)
    if theta < 0:
        theta += 2 * math.pi
    return theta

def calculate_force(input, m_c, m_f, m_p, k_c1, k_f1, k_p1, K, machine_coef_x, machine_coef_y, machine_coef_z) -> tuple:
    epsilon = 0.25
    # Konstanten
    digits = 3
    tool_radius = 5 # tool
    a_e = 2 * tool_radius # tool -> Länge des tools der Fräst bei uns der Durchmesser
    tooth_amount = 4 # tool
    kappa = math.radians(90)  # kappe = 90 da Umfangfräsen
    phi_s = math.radians(180)  # Schnittwinkel beim volle fraesen ist das 180 Grad

    # Input
    v_x, v_y, v_z, v_sp, a_p = input
    v_ges = math.sqrt(v_x**2 + v_y**2 + v_z**2)
    v_sp = v_sp * 10 / (60 * 60)    # umrechnung auf m/s
    v_x, v_y, v_z, v_sp = v_x, v_y, v_z, v_sp #umrechnung auf m/S

    if abs(v_sp) < epsilon or math.sqrt(v_x**2 + v_y**2) < epsilon:
        return 0, 0, 0, 0 

    theta = calculate_theta(v_x, v_y)

    force_rotation = np.array([[math.cos(theta), math.sin(theta), 0],
                               [math.sin(theta), -math.cos(theta), 0],
                               [0, 0, 1]])

    fz = v_ges / (v_sp * tooth_amount)  # Vorschub pro Zahn

    h = 114.6 / math.degrees(phi_s) * fz * (a_e / (tool_radius * 2)) * math.sin(kappa)  # Formel 3.11 nach Neugebauer seite 41
    b = a_p / math.sin(kappa)
    h = max(h, 0)

    F_c = b * h ** (1 - m_c) * k_c1 * K
    F_cn = b * h ** (1 - m_f) * k_f1 * K
    F_pz = b * h ** (1 - m_p) * k_p1 * K

    F_c_matrix = np.array([[F_c], [F_cn], [F_pz]])

    f_xyz = force_rotation @ F_c_matrix

    if UNIFIED_SCALE:
        f_x = f_xyz[0, 0]# * machine_coef_x
        f_y = f_xyz[1, 0]# * machine_coef_y
        f_z = f_xyz[2, 0]# * machine_coef_z
    else:
        f_x = f_xyz[0, 0] * machine_coef_x
        f_y = f_xyz[1, 0] * machine_coef_y
        f_z = f_xyz[2, 0] * machine_coef_z

    f_sp = math.sqrt(f_x ** 2 + f_y ** 2)

    return round(f_x, digits), round(f_y, digits), round(f_z, digits), round(f_sp, digits)

def objective_function(params, x, y):

    forces = [calculate_force(row, *params) for _, row in x.iterrows()]

    predicted_forces = np.array(forces)

    error = np.sum(10*(predicted_forces[:, 0] - y['f_x'].values) ** 2
                   + 10*(predicted_forces[:, 1] - y['f_y'].values) ** 2
                   + (predicted_forces[:, 2] - y['f_z'].values) ** 2
                   + 10*(predicted_forces[:, 3] - y['f_sp'].values) ** 2)
    print(error)
    return error

def objective_function_S235JR(params, x, y):

    params_f = np.array([0.17, *params[:2], 1780, *params[2:]])# m_c ist bekannt # k_c1 ist bekannt
    return objective_function(params_f, x, y)

def objective_function_with_constraints(params, x, y, constraints, penalty_weight=1000, penalty_weight_c=100, penalty_weight_m=100):
    forces = [calculate_force(row, *params) for _, row in x.iterrows()]
    predicted_forces = np.array(forces)
    error = np.sum(
        (predicted_forces[:, 0] - y['f_x'].values) ** 2 +
        (predicted_forces[:, 1] - y['f_y'].values) ** 2 +
        (predicted_forces[:, 2] - y['f_z'].values) ** 2 +
        (predicted_forces[:, 3] - y['f_sp'].values) ** 2
    )

    # Strafterm für Constraints
    penalty = 0
    m_c, m_f, m_p, k_c1, k_f1, k_p1, K, machine_koeff_x, machine_koeff_y, machine_koeff_z = params 

    if m_c < constraints['m_c_min'] or m_c > constraints['m_c_max']:
        penalty += penalty_weight_c * penalty_weight * penalty_weight_m * (max(constraints['m_c_min'] - m_c, 0) + max(m_c - constraints['m_c_max'], 0))**2  # m_c und k_c sind relativ sicher
    if m_f < constraints['m_f_min'] or m_f > constraints['m_f_max']:
        penalty += penalty_weight * penalty_weight_m * (max(constraints['m_f_min'] - m_f, 0) + max(m_f - constraints['m_f_max'], 0))**2 
    if m_p < constraints['m_p_min'] or m_p > constraints['m_p_max']:
        penalty += penalty_weight * penalty_weight_m * (max(constraints['m_p_min'] - m_p, 0) + max(m_p - constraints['m_p_max'], 0))**2 
    if k_c1 < constraints['kc1.1_min'] or k_c1 > constraints['kc1.1_max']:
        penalty += penalty_weight_c * penalty_weight * (max(constraints['kc1.1_min'] - k_c1, 0) + max(k_c1 - constraints['kc1.1_max'], 0))**2  # m_c und k_c sind relativ sicher
    if k_f1 < constraints['kf1.1_min'] or k_f1 > constraints['kf1.1_max']:
        penalty += penalty_weight * (max(constraints['kf1.1_min'] - k_f1, 0) + max(k_f1 - constraints['kf1.1_max'], 0))**2
    if k_p1 < constraints['kp1.1_min'] or k_p1 > constraints['kp1.1_max']:
        penalty += penalty_weight * (max(constraints['kp1.1_min'] - k_p1, 0) + max(k_p1 - constraints['kp1.1_max'], 0))**2
    if K < constraints['K_min'] or K > constraints['K_max']:
        penalty += penalty_weight * (max(constraints['K_min'] - K, 0) + max(K - constraints['K_max'], 0))**2

    total_error = error + penalty
    print(f"Error: {error}, Penalty: {penalty}, Total: {total_error}")
    return total_error

def filter_data(raw_data, tool_radius, part_position, part_dimension):
    mean_v_sp = raw_data['v_sp'].mean()
    std_v_sp = raw_data['v_sp'].std()

    # Festlegung der Grenzen für die Filterung
    lower_bound = mean_v_sp - 4 * std_v_sp
    upper_bound = mean_v_sp + 4 * std_v_sp

    # Filtern der Daten
    filtered_data = raw_data[abs(raw_data['v_z']) < 0.25].reset_index(drop=True)
    mask = is_tool_in_part(filtered_data, tool_radius, part_position, part_dimension)
    filtered_data = filtered_data[mask].reset_index(drop=True)
    #filtered_data = filtered_data[abs(filtered_data['f_x']) > 0.8].reset_index(drop=True)
    #filtered_data = filtered_data[(raw_data['v_sp'] >= lower_bound) & (raw_data['v_sp'] <= upper_bound)].reset_index(drop=True)
    #filtered_data = filtered_data[filtered_data['v_sp'] > 1].reset_index(drop=True)

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

def seperate_data(input_data):
    data = input_data.copy()
    data['f_sp'] = (data['f_x'] ** 2 + data['f_y'] ** 2)**0.5
    y = data[['f_x', 'f_y', 'f_z', 'f_sp']].copy() # Alte Kräfte sind um 200 Skaliert -> Übereinstimmung mit alten Parametern
    y.loc[:, ['f_x', 'f_z']] *= -1 # Vorzeichen vertauscht
    x = data[['v_x', 'v_y', 'v_z', 'v_sp', 'a_p']].copy()
    return x, y

def load_initial_params(material):
    # lade Parameter als initalwerte
    path_material_constant = 'Simulation\material_constant.csv'
    if material == 'S235JR':
        material_choose = 0
    elif material == 'AL2007T4':
        material_choose = 1
    else:
        material_choose = int(input("Material wählen:"))

    material_setting = pd.read_csv(path_material_constant, sep=';')

    # Initalparameter
    k_c1 = material_setting.loc[material_choose, 'k_c1']
    k_f1 = material_setting.loc[material_choose, 'k_f1']
    k_p1 = material_setting.loc[material_choose, 'k_p1']
    m_c = material_setting.loc[material_choose, 'm_c']
    m_f = material_setting.loc[material_choose, 'm_f']
    m_p = material_setting.loc[material_choose, 'm_p']
    
    # selbstgewählte Parameter funktionieren besser
    machine_coef_x = material_setting.loc[material_choose, 'machine_coef_x']
    machine_coef_y = material_setting.loc[material_choose, 'machine_coef_y']
    machine_coef_z = material_setting.loc[material_choose, 'machine_coef_z']

    if UNIFIED_SCALE:
        return  [m_c, m_f, m_p, k_c1, k_f1, k_p1, 1.5]
    
    return  [m_c, m_f, m_p, k_c1, k_f1, k_p1, 1.5, machine_coef_x, machine_coef_y, machine_coef_z]

def load_material_constraints(material, constraints_file='Simulation\material_parameter_constraints.csv'):
    constraints = pd.read_csv(constraints_file, sep=';')
    material_row = constraints[constraints.iloc[:, 0] == material].iloc[0]
    return {
        'kc1.1_min': material_row['kc1.1_min'],
        'm_c_min': material_row['m_c_min'],
        'kf1.1_min': material_row['kf1.1_min'],
        'm_f_min': material_row['m_f_min'],
        'kp1.1_min': material_row['kp1.1_min'],
        'm_p_min': material_row['m_p_min'],
        'K_min': material_row['K_min'],
        'kc1.1_max': material_row['kc1.1_max'],
        'm_c_max': material_row['m_c_max'],
        'kf1.1_max': material_row['kf1.1_max'],
        'm_f_max': material_row['m_f_max'],
        'kp1.1_max': material_row['kp1.1_max'],
        'm_p_max': material_row['m_p_max'],
        'K_max': material_row['K_max'],
    }

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

    elif material == "AL" and geometry[:2] == ["2007", "T4"] or material == "AL2007T4":
        material = "AL2007T4"
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

def save_named_optimized_parameters(material: str, params: list, filename='Simulation\optimized_parameters.json'): #Simulation , scale=200
    """
    Speichert benannte optimierte Parameter für ein Material als JSON-Datei.

    :param material: Name des Materials
    :param params: Liste der optimierten Parameter (Reihenfolge muss stimmen)
    :param filename: Pfad zur JSON-Datei
    """

    param_keys = [
        "m_c", "m_f", "m_p",
        "k_c1", "k_f1", "k_p1",
        "K",
        "machine_coef_x", "machine_coef_y", "machine_coef_z",
        #"t_x", "t_y", "t_z"
    ]

    if len(params) != len(param_keys):
        raise ValueError(f"Erwarte {len(param_keys)} Parameter, aber {len(params)} erhalten.")

    param_dict = dict(zip(param_keys, params))
    param_dict['scale'] = scale 
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

def correct_fz_drift(data, time_column='time', fz_column='f_z', threshold_min=0.1, threshold_max = 1, show_plots = False):
    """
    Korrigiert den Sensor-Drift in den f_z-Daten, aber nur für Werte, deren Absolutwert > threshold ist
    und die nicht mehr als 4 Standardabweichungen vom Mittelwert abweichen.

    Parameter:
    - data: DataFrame mit den Daten
    - time_column: Name der Spalte für die Zeitachse (Standard: 'time')
    - fz_column: Name der Spalte mit den f_z-Daten (Standard: 'f_z')
    - threshold: Schwellenwert für den Absolutwert von f_z (Standard: 10)

    Rückgabe:
    - DataFrame mit korrigierten f_z-Daten (neue Spalte: 'f_z_corrected')
    - Das angepasste LinearRegression-Modell
    """
    #n = int(len(data) *1/3)
    #data_filtered = data.copy().iloc[2*n:]
    # Filtere die Daten für die Regression: nur f_z mit |f_z| > threshold
    mask_abs = (np.abs(data['f_x']) < threshold_min) & (np.abs(data['f_y']) < threshold_min) & (np.abs(data[fz_column]) < threshold_max)

    # Falls keine Daten die Kriterien erfüllen, gebe die ursprünglichen Daten zurück
    if mask_abs.sum() == 0:
        print("Keine f_z-Werte erfüllen die Kriterien (|f_z| > threshold). Keine Korrektur möglich.")
        data['f_z_corrected'] = data[fz_column]
        return data['f_z_corrected'], None

    # Daten für die Regression
    X_filtered = data.loc[mask_abs, time_column].values.reshape(-1, 1)
    y_filtered = data.loc[mask_abs, fz_column].values

    # Lineare Regression anpassen (nur für gefilterte Daten)
    model = LinearRegression()
    model.fit(X_filtered, y_filtered)

    # Vorhersage der Geraden für alle Zeitpunkte
    X_all = data[time_column].values.reshape(-1, 1)
    y_pred_all = model.predict(X_all)

    # Korrigiere nur die f_z-Werte, die die Kriterien erfüllen
    fz_corrected = data[fz_column].copy() - (model.predict(np.array(data[time_column]).reshape(-1, 1))- model.intercept_)

    # Füge die korrigierten Daten als neue Spalte hinzu
    data = data.copy()
    data['f_z_corrected'] = fz_corrected

    if show_plots:
        # Plotten der Daten
        plt.figure(figsize=(12, 6))

        # Ursprüngliche Daten
        plt.plot(
            data[time_column],
            data[fz_column], #fz_column
            label='Original f_z',
            color='blue',
            alpha=0.7,
            linewidth=1.5
        )

        # Korrigierte Daten
        plt.plot(
            data[time_column],
            data['f_z_corrected'],
            label='Korrigiertes f_z',
            color='green',
            alpha=0.7,
            linewidth=1.5
        )

        # Angepasste Gerade (nur für gefilterte Daten)
        plt.plot(
            data[time_column],
            y_pred_all,
            label='Angepasste Gerade (Drift, gefiltert)',
            color='red',
            linestyle='--',
            linewidth=1.5
        )

        # Beschriftungen und Legende
        plt.xlabel('Zeit')
        plt.ylabel('f_z')
        plt.title(f'Korrektur des Sensor-Drifts in f_z (|f_z| > {threshold_min}, |f_z| < {threshold_max})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # Anzeigen des Plots
        plt.show()

    return data['f_z_corrected'], model

def objective_scale(params, x, y):
    
    error = np.sum(10*(x[:, 0] - y['f_x'].values * params) ** 2
                   + 10*(x[:, 1] - y['f_y'].values * params) ** 2
                   + (x[:, 2] - y['f_z'].values * params) ** 2
                   + 10*(x[:, 3] - y['f_sp'].values * params) ** 2)
    print(error)
    return error

tool_radius = 5

if __name__ == '__main__':

    TRAININGSDATA = 'Notch_Normal'
    FIXED_PARAMETER = True
    USE_REGULARISATION = False
    UNIFIED_SCALE = False
    PLOT = True
    
    EXCLUDE = None

    SCALE = 200 # Skalieriert die Kräfte auf eine ähnliche größenordnung wie die Uhrsprünglichen Simulationsdaten

    x_s = []
    y_s = []
    scale = 1

    if UNIFIED_SCALE:
        # Optimierung der Skalierung
        for material in ['S235JR', 'AL2007T4']:

            print('load data')
            path = 'DataSets_CMX_Plate_Notch_Gear\DataMerged'
            all_files = glob.glob(os.path.join(path, f'{material}*.csv'))
            if EXCLUDE != None:            
                exclude_files = {f'{material}_{EXCLUDE}.csv'}
                files = [os.path.basename(f) for f in all_files if os.path.basename(f) not in exclude_files]
            else:
                files = [os.path.basename(f) for f in all_files]
          

            x_all = []
            y_all = []
            for file in files:
                #if not ('Depth' in file or 'Normal' in file): # Reduzierte Daten, verbessert Performance, bei Gear SF kam es zum ausbruch daher ausschluss von SF. Depth ist am wichtigsten
                #    continue
                path_data = os.path.join(path, file)
                raw_data = pd.read_csv(path_data)

                if not TRAININGSDATA in file:
                    continue

                if file.startswith('S235'):                
                    if file.endswith('Gear_SF.csv'): # bei Gear SF kam es zum ausbruch daher ausschluss 
                        continue

                if file.startswith('AL'):
                    
                    if file.endswith('Depth.csv'):
                        print('Schneide Fehlerhafte bereiche raus')
                        n = int(2/3 *len(raw_data))
                        raw_data = raw_data.iloc[:n].reset_index(drop=True)

                    print('Korrigiere f_z')

                    # Korrigiere f_z
                    raw_data['f_z'], model = correct_fz_drift(raw_data, show_plots=False)

                _, part_position, part_dimension = get_part_properties(file)

                raw_data['a_p'] = calculate_a_p(raw_data['pos_z'], part_position[2], part_dimension[2])

                filtered_data = filter_data(raw_data, tool_radius, part_position, part_dimension)
                
                x, y = seperate_data(filtered_data)
                x_all.append(x.reset_index(drop=True))
                y_all.append(y.reset_index(drop=True))

            x = pd.concat(x_all, ignore_index=True)
            y = pd.concat(y_all, ignore_index=True)
            y = y * SCALE

            initial_params = load_initial_params(material)
            prams = initial_params
            
            # Korrektur für S235JR
            if material == 'S235JR':
                [m_c, m_f, m_p, k_c1, k_f1, k_p1, K]= initial_params
                params = [0.17,m_f, m_p, 1780,  k_f1, k_p1, K] 
                initial_params = [m_f, m_p, k_f1, k_p1, K] 
            f_sim = [calculate_force(row, *params) for _, row in x.iterrows()]

            y_s.append(y)
            x_s.append(f_sim)

        options = {
            'maxiter': 50000,
            'fatol': 1e-3,
            'xatol': 1e-3,
            'disp': True
        }
            
        # skalierung bestimmen.
        f_sim = np.vstack(x_s)
        y =pd.concat(y_s, axis=0)

        result = minimize(
            objective_scale,
            1,
            args=(f_sim, y), #args=(x, y, constraints),
            method='L-BFGS-B',
            options=options
        )

        scale = result.x[0] 
        print(scale)
    
    for material in ['S235JR', 'AL2007T4']:

        print('load data')
        path = 'DataSets_CMX_Plate_Notch_Gear\DataMerged'
        all_files = glob.glob(os.path.join(path, f'{material}*.csv'))
        
        if EXCLUDE != None:            
            exclude_files = {f'{material}_{EXCLUDE}.csv'}
            files = [os.path.basename(f) for f in all_files if os.path.basename(f) not in exclude_files]
        else:
            files = [os.path.basename(f) for f in all_files]


        x_all = []
        y_all = []
        for file in files:
            #if not ('Depth' in file or 'Normal' in file): # Reduzierte Daten, verbessert Performance, bei Gear SF kam es zum ausbruch daher ausschluss von SF. Depth ist am wichtigsten
            #    continue
            path_data = os.path.join(path, file)
            raw_data = pd.read_csv(path_data)

            if not TRAININGSDATA in file:
                continue

            if file.startswith('S235'):                
                if file.endswith('Gear_SF.csv'): # bei Gear SF kam es zum ausbruch daher ausschluss 
                    continue

            if file.startswith('AL'):
                
                if file.endswith('Depth.csv'):
                    print('Schneide Fehlerhafte bereiche raus')
                    n = int(2/3 *len(raw_data))
                    raw_data = raw_data.iloc[:n].reset_index(drop=True)

                print('Korrigiere f_z')

                # Korrigiere f_z
                raw_data['f_z'], model = correct_fz_drift(raw_data, show_plots=PLOT)

            _, part_position, part_dimension = get_part_properties(file)

            raw_data['a_p'] = calculate_a_p(raw_data['pos_z'], part_position[2], part_dimension[2])

            filtered_data = filter_data(raw_data, tool_radius, part_position, part_dimension)
            
            x, y = seperate_data(filtered_data)
            x_all.append(x.reset_index(drop=True))
            y_all.append(y.reset_index(drop=True))

        x = pd.concat(x_all, ignore_index=True)
        y = pd.concat(y_all, ignore_index=True)
        y = y * SCALE * scale
        
        if PLOT:
            plt.figure(figsize=(12, 8))
            plt.plot(y['f_x'], label='f_x')
            plt.legend()
            plt.show()

        initial_params = load_initial_params(material)
        prams = initial_params

        if FIXED_PARAMETER:
        # Korrektur für S235JR
            if material == 'S235JR':
                [m_c, m_f, m_p, k_c1, k_f1, k_p1, K, machine_coef_x, machine_coef_y, machine_coef_z]= initial_params  # 
                params = [0.17, m_f, m_p, 1780,  k_f1, k_p1, K, machine_coef_x, machine_coef_y, machine_coef_z] 
                initial_params = [m_f, m_p, k_f1, k_p1, K, machine_coef_x, machine_coef_y, machine_coef_z] 

        options = {
            'maxiter': 50000,
            'fatol': 1e-3,
            'xatol': 1e-3,
            'disp': True
        }
                 
        if FIXED_PARAMETER:
            if material == 'S235JR':
                ob_func = objective_function_S235JR
            else:
                ob_func = objective_function
        else:
            if USE_REGULARISATION:
                ob_func = objective_function_with_constraints

                constraints = load_material_constraints(material)

                result = minimize(
                    ob_func, #objective_function_with_constraints,
                    initial_params,
                    args=(x, y, constraints),
                    method='Nelder-Mead',
                    options=options
                )
            else:
                ob_func = objective_function

        if not USE_REGULARISATION:           
            result = minimize(
                ob_func,
                initial_params,
                args=(x, y), 
                method='Nelder-Mead',
                options=options
            )
        optimized_params = result.x

        if FIXED_PARAMETER:
            if material == 'S235JR':
                optimized_params = result = np.array([0.17, *optimized_params[:2], 1780, *optimized_params[2:]])
        
        print(optimized_params)

        if UNIFIED_SCALE:
            params = optimized_params
            optimized_params = np.array([*optimized_params, 1/scale, 1/scale, 1/scale])

        save_named_optimized_parameters(material, optimized_params.tolist())

        if UNIFIED_SCALE:
            optimized_params = params

        if PLOT:

            forces = [calculate_force(row, *optimized_params) for _, row in x.iterrows()]
            forces = np.array(forces)
            for i, axis in enumerate(['x', 'y', 'z']):
                plot_forces(forces, y, i, axis, ref_forces=None)

            print('Starte Test')
            for geometry in ['Plate', 'Gear']:
                file = f'{material}_{geometry}_Normal.csv'
                path_data = os.path.join(path, file)
                raw_data = pd.read_csv(path_data)
                _, part_position, part_dimension = get_part_properties(file)
                raw_data['a_p'] = calculate_a_p(raw_data['pos_z'], part_position[2], part_dimension[2])

    
                plt.figure(figsize=(12, 8))
                plt.plot(raw_data['a_p'], label='a_p')
                plt.legend()
                plt.show()

                x, y = seperate_data(raw_data)
                y = y * scale * SCALE
                forces = [calculate_force(row, *optimized_params) for _, row in x.iterrows()]
                forces = np.array(forces)
                for i, axis in enumerate(['x', 'y', 'z']):
                    plot_forces(forces, y, i, axis, ref_forces=None)