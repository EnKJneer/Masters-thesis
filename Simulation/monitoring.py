import numpy as np
import pandas as pd
import machine_state as ms
import value_plot as vp
from MMR_Calculator import voxel_class_numba as vc, voxel_image, a_p_calculator, Optimized_Calculation as OC
from Simulation.MMR_Calculator.MRRSimulation import CNCMRRSimulation


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

def state_monitoring(machine_state: ms.MachineState, a_p_change: bool, tool: vc.Tool, part: vc.PartialVoxelPart,
                     process_data: pd.DataFrame, true_curr: pd.DataFrame, frequence: int,
                     circle_part: list, mrr_monitoring: bool = False) -> pd.DataFrame:

    a_e = machine_state.get_tool_radius() * 2
    epsilon = 1e-3
    digits = 3
    force_df = pd.DataFrame()
    new_part_volume = part.get_total_volume()
    print(f'Volumen vorher: {new_part_volume}')

    #calculator = OC.OptimizedMRRCalculator()
    #mrr_results = calculator.calculate_mrr_optimized(process_data, part, tool, frequence)
    #part_position = part.coordinates
    #part_dimension = [part.width, part.depth, part.height, part.voxel_size]
    #simulator = CNCMRRSimulation(part_position, part_dimension, tool.radius)
    #times, mrr_values = simulator.simulate_mrr(process_data)
    #simulator.plot_results(times, mrr_values, process_data)

    for i in range(process_data.shape[0]):
        a_p = a_p_calculator.a_p_cal(i, a_p_change)
        pos_x = process_data.loc[i, 'pos_x']
        pos_y = process_data.loc[i, 'pos_y']
        pos_z = process_data.loc[i, 'pos_z']
        new_tool_coordinates = [pos_x, pos_y, pos_z]
        v_x = process_data.loc[i, 'v_x']
        v_y = process_data.loc[i, 'v_y']
        v_z = process_data.loc[i, 'v_z']
        v_sp = process_data.loc[i, 'v_sp']

        current_process_state = ms.ProcessState(pos_x, pos_y, pos_z, v_x, v_y, v_z, v_sp, a_p, a_e)
        tool.set_new_position(new_tool_coordinates)

        '''
        if abs(v_sp) < epsilon:
            materialremoved_sim = 0
        else:
            old_part_volume = new_part_volume
            part.apply_tool(tool)
            new_part_volume = part.get_total_volume()
            if old_part_volume != new_part_volume:
                materialremoved_sim = round((old_part_volume - new_part_volume) * frequence, digits)


            elif old_part_volume < new_part_volume:
                raise ValueError("Alte Volumen ist groesser als neue Volume!")
            else:
                materialremoved_sim = 0'''
        #materialremoved_sim = mrr_values[i]
        #materialremoved_sim = mrr_results[i]
        z_rel = round(tool.z_position - part.z_origin - part.height, digits)
        if z_rel < 0:
            f_x_sim, f_y_sim, f_z_sim, f_sp_sim = force_calculate(machine_state, current_process_state, frequence)
            #f_x_sim = round(np.clip(f_x_sim, -1000, 1000) / 200, digits)
            #f_y_sim = round(np.clip(f_y_sim, -1000, 1000) / 200, digits)
            #f_z_sim = round(np.clip(f_z_sim, -1000, 1000) / 200, digits)
            #f_sp_sim = round(np.clip(f_sp_sim, -1000, 1000) / 200, digits)
        else:
            f_x_sim, f_y_sim, f_z_sim, f_sp_sim = 0.0, 0.0, 0.0, 0.0
        status = round(i / process_data.shape[0] * 100, 2)
        print(f'Satus: {status} f_x_sim: {f_x_sim}')
        force_df = pd.concat([force_df, test_update_force_df(f_x_sim, f_y_sim, f_z_sim, f_sp_sim, 0)], ignore_index=True)

    data_df = pd.concat([process_data, true_curr, force_df], axis=1)
    data_df = data_df.rename(columns={'pos_x': 'pos_x', 'pos_y': 'pos_y', 'pos_z': 'pos_z', 'MRR': 'materialremoved_sim',
                                      'f_x_sim': 'f_x_sim', 'f_y_sim': 'f_y_sim', 'f_z_sim': 'f_z_sim', 'f_sp_sim': 'f_sp_sim'})

    print(f'Volumen nach der Bearbeitung: {part.get_total_volume()}')
    return data_df

def force_calculate(machine_state: ms.MachineState, process_state: ms.ProcessState, frequence: int) -> tuple:
    return process_state.calculate_force(machine_state, frequence)

def test_update_force_df(f_x, f_y, f_z, f_sp, mrr) -> pd.DataFrame:
    temp_force_df = pd.DataFrame([[f_x, f_y, f_z, f_sp, mrr]], columns=['f_x_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim', 'MRR'])
    return temp_force_df