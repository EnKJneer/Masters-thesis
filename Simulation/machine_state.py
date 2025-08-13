import json

import numpy as np
import math
import pandas as pd
from MMR_Calculator import voxel_class_numba

class MachineState:
    def __init__(self, k_c1, k_f1, k_p1, K_v, K_kss, x: float, y: float, z: float, tool_radius: float, tooth_amount: int, machine_coef_x: float, machine_coef_y: float, machine_coef_z: float) -> None:
        self.k_c1 = k_c1
        self.k_f1 = k_f1
        self.k_p1 = k_p1
        self.K_v = K_v
        self.K_kss = K_kss
        self.x = x
        self.y = y
        self.z = z
        self.tool_radius = tool_radius
        self.tooth_amount = tooth_amount
        self.machine_coef_x = machine_coef_x
        self.machine_coef_y = machine_coef_y
        self.machine_coef_z = machine_coef_z
        self.teeth = [i * (2 * math.pi / self.tooth_amount) for i in range(self.tooth_amount)]

    def get_tool_radius(self) -> float:
        return self.tool_radius

    def set_tool_radius(self, new_radius) -> None:
        self.tool_radius = new_radius

    def set_theeth_angle(self, n: float) -> None:
        for i in range(len(self.teeth)):
            self.teeth[i] = (self.teeth[i] + n) % (2 * math.pi)

class ProcessState:
    def __init__(self, x_pos: float, y_pos: float, z_pos: float, v_x: float, v_y: float, v_z: float, v_sp: float, a_p: float, a_e: float, phi: float) -> None:
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.z_pos = z_pos
        self.v_x = v_x
        self.v_y = v_y
        self.v_z = v_z
        self.v_c = math.sqrt(v_x**2 + v_y**2)
        self.v_ges = math.sqrt(v_x**2 + v_y**2 + v_z**2)
        self.process_null_state = v_sp == 0 or self.v_c == 0
        self.theta = self.calculate_theta(v_x, v_y)
        self.v_sp = v_sp
        self.a_e = a_e
        self.a_p = a_p
        self.phi = phi

    def calculate_theta(self, v_x, v_y):
        if v_x == 0 and v_y == 0:
            return 0
        theta = math.atan2(v_y, v_x)
        if theta < 0:
            theta += 2 * math.pi
        return theta

    def calculate_force(self, machine_state: MachineState, frequence: int) -> tuple:
        digits = 3
        if not self.process_null_state:
            theta = self.theta
            a_p = self.a_p
            a_e = self.a_e
            v_ges = self.v_ges
            v_sp = self.v_sp * 10 / (60*60)
            v_x = self.v_x
            v_y = self.v_y
            v_z = self.v_z
            x = machine_state.x
            y = machine_state.y
            z = machine_state.z
            k_c1 = machine_state.k_c1
            k_f1 = machine_state.k_f1
            k_p1 = machine_state.k_p1
            K_v = machine_state.K_v
            K_kss = machine_state.K_kss
            tool_radius = machine_state.tool_radius
            tooth_amount = machine_state.tooth_amount
            kappa = math.radians(90) # kappe = 90 da Umfangfräsen
            phi_s = math.radians(self.phi) #Schnittwinkel beim volle fraesen ist das 180 Grad
            teeth = machine_state.teeth
            machine_coef_x = machine_state.machine_coef_x
            machine_coef_y = machine_state.machine_coef_y
            machine_coef_z = machine_state.machine_coef_z

            force_rotation = np.array([[math.cos(theta), math.sin(theta), 0],
                                       [math.sin(theta), -math.cos(theta), 0],
                                       [0, 0, 1]])

            fz = v_ges / (v_sp * tooth_amount) # Vorschub pro Zahn
            angle_update = v_sp / (60 * frequence) * (2*math.pi)
            machine_state.set_theeth_angle(angle_update)
            h = 114.6 / math.degrees(phi_s) * fz * (a_e / (tool_radius * 2)) * math.sin(kappa) # Formel 3.11 nach Neugebauer seite 41
            b = a_p / math.sin(kappa)
            h = max(h, 0)

            F_c = b * h ** (1-z) * k_c1 * K_v * K_kss
            F_cn = b * h ** (1-x) * k_f1 * K_v * K_kss
            F_pz = b * h ** (1-y) * k_p1 * K_v * K_kss

            F_c_matrix = np.array([[F_c], [F_cn], [F_pz]])

            #F_fz = np.array([F_c * math.cos(teeth[i]) + F_cn * math.sin(teeth[i]) if teeth[i] <= phi_s else 0 for i in range(len(teeth))])
            #F_fnz = np.array([F_c * math.sin(teeth[i]) - F_cn * math.cos(teeth[i]) if teeth[i] <= phi_s else 0 for i in range(len(teeth))])

            f_xyz = force_rotation @ F_c_matrix
            f_x = f_xyz[0,0] * machine_coef_x
            f_y = f_xyz[1,0] * machine_coef_y
            f_z = f_xyz[2,0] * machine_coef_z
            f_sp = math.sqrt(f_x**2 + f_y**2)
        else:
            f_x, f_y, f_z, f_sp = 0, 0, 0, 0

        return round(f_x, digits), round(f_y, digits), round(f_z, digits), round(f_sp, digits)


def read_machine_state(data_path: str) -> pd.DataFrame:
    state_df = pd.read_csv(data_path, sep=';')

    return state_df

def load_optimized_parameters_as_dataframe(json_path: str) -> pd.DataFrame:
    """
    Lädt optimierte Parameter aus einer JSON-Datei und gibt sie als DataFrame zurück,
    formatiert wie: Material;k_c1;k_f1;k_p1;x;y;z;machine_coef_x;machine_coef_y;machine_coef_z

    :param json_path: Pfad zur JSON-Datei mit den Parametern
    :return: Pandas DataFrame mit den Parametern pro Material
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    rows = []
    for material, params in data.items():
        row = {
            'Material': material,
            'k_c1': params['k_c1'],
            'k_f1': params['k_f1'],
            'k_p1': params['k_p1'],
            'x': params['x'],
            'y': params['y'],
            'z': params['z'],
            'machine_coef_x': params['machine_coef_x'],
            'machine_coef_y': params['machine_coef_y'],
            'machine_coef_z': params['machine_coef_z']
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=[
        'Material', 'k_c1', 'k_f1', 'k_p1', 'x', 'y', 'z',
        'machine_coef_x', 'machine_coef_y', 'machine_coef_z'
    ])
    return df

def load_optimized_parameters_as_dict(json_path: str) -> dict:
    """
    Lädt optimierte Parameter aus einer JSON-Datei und gibt sie als Dictionary zurück,
    wobei nur die relevanten Parameter enthalten sind (ohne 'K').

    :param json_path: Pfad zur JSON-Datei mit den Parametern
    :return: Dictionary im Format {Material: {k_c1, k_f1, k_p1, x, y, z, machine_coef_x, ...}}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = {}

    for material, params in data.items():
        result[material] = {
            'k_c1': params['k_c1'],
            'k_f1': params['k_f1'],
            'k_p1': params['k_p1'],
            'x': params['x'],
            'y': params['y'],
            'z': params['z'],
            'machine_coef_x': params['machine_coef_x'],
            'machine_coef_y': params['machine_coef_y'],
            'machine_coef_z': params['machine_coef_z'],
            'K': params['K']
        }

    return result

def set_machine_state(setting_dict: dict, material: str, tool_diameter: float,
                      position: list[float], dimension: list[float]) -> tuple[
    MachineState, voxel_class_numba.Tool, voxel_class_numba.PartialVoxelPart]:

    # Festdefinierte Werte
    K_v = 1  # Verschleißkorrekturfaktor
    K_kss = 0.9
    tool_radius = tool_diameter / 2
    tooth_amount = 4

    # Startposition von Tool und Werkstück
    x0_tool, y0_tool, z0_tool = 0, 0, 0
    x0_part, y0_part, z0_part = position
    part_width, part_depth, part_height, voxel_size = dimension

    # Parameter aus dem Dictionary laden
    if material not in setting_dict:
        raise ValueError(f"Material '{material}' nicht im Parameter-Dictionary vorhanden.")

    params = setting_dict[material]
    K_kss = params['K']
    k_c1 = params['k_c1']
    k_f1 = params['k_f1']
    k_p1 = params['k_p1']
    x = params['x']
    y = params['y']
    z = params['z']
    machine_coef_x = params['machine_coef_x']
    machine_coef_y = params['machine_coef_y']
    machine_coef_z = params['machine_coef_z']

    # Maschinenzustand erzeugen
    new_machine_state = MachineState(
        k_c1, k_f1, k_p1, K_v, K_kss,
        x, y, z,
        tool_radius, tooth_amount,
        machine_coef_x, machine_coef_y, machine_coef_z
    )

    # Werkzeug und Werkstück definieren
    tool = voxel_class_numba.Tool(tool_radius, [x0_tool, y0_tool, z0_tool])
    part = voxel_class_numba.PartialVoxelPart(part_width, part_depth, part_height, voxel_size, [x0_part, y0_part, z0_part])

    return new_machine_state, tool, part