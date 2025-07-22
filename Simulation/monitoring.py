import numpy as np
import pandas as pd
from unicodedata import digit

import machine_state as ms
import  value_plot as vp

'''from state_monitoring import (
        machine_state   as ms,
        anomaly         as ano,
        value_plot      as vp
    )'''
from MMR_Calculator import(
    voxel_class_numba   as vc,
    voxel_image,
    a_p_calculator
    )

def calculate_average (parameter_list: list, parameter_new, amount: int  = 50, mrr: bool = False) -> float:

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

def state_monitoring (machine_state: ms.MachineState, a_p_change: bool, tool: vc.Tool, part: vc.Part,
                      process_data: pd.DataFrame, true_curr: pd.DataFrame, frequence: int,
                      circle_part: list,
                      mrr_monitoring: bool = False,
                      ) -> tuple [pd.DataFrame, list[list, list, list, list], pd.DataFrame]:

    #Definition von a_p und a_e
    a_e = machine_state.get_tool_radius() * 2

    epsilon = 1e-3
    digits = 3

    #true_curr = data_preprocessing.normalize_data(true_curr)[0]

    #Definieren Vorhersagen DataFrame
    force_df = pd.DataFrame()

    #Kraft Listen
    MRR_list = []

    new_part_volume = part.get_total_volume()
    print(f'Volumen vorher: {new_part_volume}')

    for i in range(process_data.shape[0]):
        #Ermitteln a_p
        a_p = a_p_calculator.a_p_cal(i, a_p_change)

        #Position aus aktuelle Datenpunkt lesen
        pos_x = process_data.loc[i, 'pos_x']
        pos_y = process_data.loc[i, 'pos_y']
        pos_z = process_data.loc[i, 'pos_z']

        new_tool_coordinates = [pos_x, pos_y, pos_z]
        # Geschwindigkeit aus aktuelle Datenpunkt lesen
        v_x = process_data.loc[i, 'v_x']
        v_y = process_data.loc[i, 'v_y']
        v_z = process_data.loc[i, 'v_z']
        v_sp = process_data.loc[i, 'v_sp']

        #Definieren aktuelle Prozesszustand
        current_process_state = ms.ProcessState(pos_x, pos_y, pos_z, v_x, v_y, v_z, v_sp, a_p, a_e)
        #Einstellung aktuelle Position von Tool
        tool.set_new_position(new_tool_coordinates)

        #Berechnen von MRR
        if abs(v_sp) < epsilon:
            materialremoved_sim = 0

        else:
            #print(voxel_image.find_false_coordinates(part.voxels_fill))             #Z-Achse Koordinate nicht bekannt dann diese Zeile ausfuehren um rauszu finden welche Ebene abgertragen wird!
            old_part_volume = new_part_volume
            part.apply_tool(tool)
            new_part_volume = part.get_total_volume()

            if old_part_volume != new_part_volume:

                materialremoved_sim = round((old_part_volume - new_part_volume) * frequence, digits)

                #print(f'MRR_{i}: {materialremoved_sim}')

            elif old_part_volume < new_part_volume:

                raise ValueError("Alte Volumen ist groesser als neue Volume!")

            else:
                materialremoved_sim = 0

        z_rel = round(tool.z_position - part.z_origin - part.height, digits)
        if abs(materialremoved_sim) > epsilon and z_rel < 0 and abs(materialremoved_sim) > epsilon: # ToDo: Eigentlich muss hier geprüft werden ob sich der Fräser im Eingriff befindet.
            f_x_sim, f_y_sim, f_z_sim, f_sp_sim = force_calculate(machine_state, current_process_state, frequence)
            # ToDo: Wenn die Simulation richtig funktioniert wird das nicht mehr benötigt
            f_x_sim = round(np.clip(f_x_sim, -1000, 1000) / 200, digits)
            f_y_sim = round(np.clip(f_y_sim, -1000, 1000) / 200, digits)
            f_z_sim = round(np.clip(f_z_sim, -1000, 1000) / 200, digits)
            f_sp_sim = round(np.clip(f_sp_sim, -1000, 1000) / 200, digits)
        else:
            f_x_sim, f_y_sim, f_z_sim, f_sp_sim = 0.0, 0.0, 0.0, 0.0

        print(f'z_rel: {z_rel} f_x_sim: {f_x_sim} materialremoved_sim: {materialremoved_sim}')

        #force_df = pd.concat([force_df, test_update_force_df(f_x_sim, f_y_sim, f_z_sim, f_sp_sim, MRR_average) ], ignore_index=True)
        force_df = pd.concat([force_df, test_update_force_df(f_x_sim, f_y_sim, f_z_sim, f_sp_sim, z_rel)],
                             ignore_index=True)
        if not mrr_monitoring:
            force_df = force_df.drop(columns=['MRR'])

        data_df = pd.concat([process_data, true_curr, force_df], axis=1)
        data_df = data_df.rename(columns={'pos_x':  'pos_x',
                                          'pos_y':  'pos_y',
                                          'pos_z':  'pos_z',
                                          'MRR':    'z_rel', # materialremoved_sim
                                          'f_x_sim':    'f_x_sim',
                                          'f_y_sim':    'f_y_sim',
                                          'f_z_sim':    'f_z_sim',
                                          'f_sp_sim':   'f_sp_sim',})

    print(f'Volumen nach der Bearbeitung: {part.get_total_volume()}')
    return data_df

def force_calculate (machine_state: ms.MachineState, process_state: ms.ProcessState, frequence: int) -> tuple [float, float, float, float]:

    current_process_state = process_state

    force = current_process_state.calculate_force(machine_state, frequence)

    return force

def test_update_force_df (f_x, f_y, f_z, f_sp, mrr) -> pd.DataFrame:

    temp_force_df = pd.DataFrame([[f_x, f_y, f_z, f_sp, mrr]],  columns= ['f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim', 'MRR'])

    return temp_force_df
