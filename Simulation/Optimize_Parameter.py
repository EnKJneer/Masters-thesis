import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    return round(f_x/ 200, digits), round(f_y/ 200, digits), round(f_z/ 200, digits)#, round(f_sp, digits)

def objective_function(params, x, y):
    forces = [calculate_force(row, *params) for _, row in x.iterrows()]
    predicted_forces = np.array(forces)
    error = np.sum((predicted_forces - y.values) ** 2)
    print(error)
    return error

if __name__ == '__main__':

    part_position = [-35.64, 175.0, 354.94]
    part_dimension = [75.0, 75.0*2, 50.0, 0.1]

    print('load data')
    path_data = '..\\DataSetsV3/DataMerged/S235JR_Plate_Normal.csv'
    material = 'S235JR'#'S235JR'
    raw_data = pd.read_csv(path_data)

    mean_v_sp = raw_data['v_sp'].mean()
    std_v_sp = raw_data['v_sp'].std()

    # Festlegung der Grenzen für die Filterung
    lower_bound = mean_v_sp - 4 * std_v_sp
    upper_bound = mean_v_sp + 4 * std_v_sp

    # Filtern der Daten
    filtered_data = raw_data[(raw_data['v_sp'] >= lower_bound) & (raw_data['v_sp'] <= upper_bound)].reset_index(drop=True)
    filtered_data = filtered_data[filtered_data['v_sp'] > 1].reset_index(drop=True)
    filtered_data = filtered_data[abs(filtered_data['v_z']) < 0.25].reset_index(drop=True)
    # ToDo: Datenrasufiltern wo das tool nicht im Bauteil ist

    # Berechne a_p
    filtered_data['a_p'] = 6 * np.ones(len(filtered_data.index))   #round(raw_data['pos_z'] - part_position[2] - part_dimension[2], 3)

    y = filtered_data[['f_x', 'f_y', 'f_z']]
    y = y *-1
    x = [v_x, v_y, v_z, v_sp, a_p] = filtered_data[['v_x', 'v_y', 'v_z', 'v_sp', 'a_p']]

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
    machine_coef_x = material_setting.loc[material_choose, 'machine_coef_x']
    machine_coef_y = material_setting.loc[material_choose, 'machine_coef_y']
    machine_coef_z = material_setting.loc[material_choose, 'machine_coef_z']

    initial_params = [x_param, y_param, z_param, k_c1, k_f1, k_p1, 0.9, machine_coef_x, machine_coef_y, machine_coef_z]

    # 1. Optimiere Parameter
    # Optionen für die Optimierung
    options = {
        'maxiter': 500,  # Maximale Anzahl von Iterationen
        'fatol': 100,  # Toleranz für Änderungen in der Zielfunktion
        'xatol': 1e-6,  # Toleranz für Änderungen in den Parametern
        'disp': True  # Fortschrittsinformationen anzeigen
    }

    result = minimize(objective_function, initial_params, args=(x, y), method='SLSQP', options=options) #SLSQP Nelder-Mead
    optimized_params = result.x
    print(optimized_params)

    # 2. Berechne Kräfte mit optimierten Parametern
    forces = [calculate_force(row, *optimized_params) for _, row in x.iterrows()]
    forces = np.array(forces)
    forces[:, 0] = np.clip(forces[:, 0], -5, 5)
    # 3. Plotte Kräfte
    plt.figure(figsize=(12, 8))
    plt.plot(forces[:, 0], label='Force X')
    plt.plot(y['f_x'], label='gt X')
    #plt.plot(forces[:, 1], label='Force Y')
    #plt.plot(forces[:, 2], label='Force Z')
    plt.legend()
    plt.title('Calculated Forces')
    plt.xlabel('Sample')
    plt.ylabel('Force')
    plt.show()