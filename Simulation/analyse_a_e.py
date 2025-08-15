import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('TkAgg')

def calculate_ae_and_angle_for_tool(tool_x, tool_y, tool_z, tool_radius, part_position, part_dimension):
    radius = tool_radius
    D = 2.0 * radius

    x_min = part_position[0]
    y_min = part_position[1]
    z_min = part_position[2]
    x_max = x_min + part_dimension[0]
    y_max = y_min + part_dimension[1]
    z_max = z_min + part_dimension[2]

    # Wenn der Bohrer komplett über dem Werkstück ist → kein Eingriff
    if tool_z <= z_min or tool_z >= z_max:
        return 0.0, 0.0

    # Vollständig draußen in XY
    if (tool_x + radius <= x_min or tool_x - radius >= x_max or
        tool_y + radius <= y_min or tool_y - radius >= y_max):
        return 0.0, 0.0

    # Vollständig drin in XY
    if (tool_x - radius >= x_min and tool_x + radius <= x_max and
        tool_y - radius >= y_min and tool_y + radius <= y_max):
        return D, 180.0

    # Teilweise drin → max. Schnittlänge a_e bestimmen
    max_chord = 0.0
    step = max(radius / 180.0, 1e-6)
    for y in frange(max(y_min, tool_y - radius), min(y_max, tool_y + radius), step):
        dy = y - tool_y
        r2_minus_dy2 = radius * radius - dy * dy
        if r2_minus_dy2 <= 0:
            continue
        dx = math.sqrt(r2_minus_dy2)
        x_left = tool_x - dx
        x_right = tool_x + dx
        chord_left = max(x_left, x_min)
        chord_right = min(x_right, x_max)
        chord_length = max(chord_right - chord_left, 0.0)
        if chord_length > max_chord:
            max_chord = chord_length

    a_e = max_chord

    # Literaturformel für Schnittwinkel
    u = max(D - a_e, 0.0)

    def clamp_cos_arg(x):
        return max(-1.0, min(1.0, x))

    c2 = clamp_cos_arg(1.0 - (2.0 * (u + a_e)) / D)
    c1 = clamp_cos_arg(1.0 - (2.0 * u) / D)

    phi2 = math.acos(c2)
    phi1 = math.acos(c1)
    phi_s = phi2 - phi1
    angle_deg = math.degrees(max(0.0, phi_s))

    return a_e, angle_deg

def frange(start, stop, step):
    v = start
    while v <= stop + 1e-12:
        yield v
        v += step

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
                part_position = [-33.15, 174.482, 354.1599]
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

path = '..\\DataSets\\DataMerged'
# Create target directory if it doesn't exist

files = os.listdir(path)
files = ['S235JR_Plate_Normal.csv']
for file in files:
    if not file.endswith('.csv'):
        continue

    print(f'Processing {file}')

    path_data = os.path.join(path, file)
    data = pd.read_csv(path_data)

    material, part_position, part_dimension = get_part_properties(file)

    tooth_amount = 4
    tool_radius = 10
    kappa = 90

    v_sp = data['v_sp'] * 10 / (60 * 60)
    v_ges = (data['v_x']**2 + data['v_y']**2)**0.5
    fz = v_ges / (v_sp * tooth_amount + 1e-1)  # Vorschub pro Zahn

    fz = np.clip(fz, -0.05, 0.05)

    a_e_array = []
    phi_s_array = []
    h_array = []

    for index, row in data.iterrows():
        a_e, phi_s = calculate_ae_and_angle_for_tool(row['pos_x'], row['pos_y'], row['pos_z'], tool_radius, part_position, part_dimension)
        phi_s = 180
        if math.degrees(phi_s) != 0:
            h = 114.6 / math.degrees(phi_s) * fz[index] * (a_e / (tool_radius * 2)) * math.sin(kappa)
        else:
            h = 0

        a_e_array.append(a_e)
        phi_s_array.append(phi_s)
        h_array.append(h)

    fig, axes = plt.subplots(4, 1, figsize=(8, 10))  # 3 Zeilen, 1 Spalte

    # Subplot für a_e_array
    axes[0].plot(a_e_array)
    axes[0].set_title('a_e')

    # Subplot für phi_s_array
    axes[1].plot(phi_s_array)
    axes[1].set_title('phi')

    # Subplot für h_array
    axes[2].plot(h_array)
    axes[2].set_title('h')

    # Subplot für f_z
    axes[3].plot(fz)
    axes[3].set_title('fz')

    # Layout anpassen, damit sich die Subplots nicht überlappen
    plt.tight_layout()
    plt.show()