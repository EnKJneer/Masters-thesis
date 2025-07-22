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

@jit
def calculate_forces(machine_state, process_states, frequence):
    def calculate_force(process_state):
        return process_state.calculate_force(machine_state, frequence)
    return vmap(calculate_force)(process_states)

if __name__ == '__main__':
    path_data = '..\\DataSetsV3/DataMerged/S235JR_Plate_Normal.csv'
    material = 'S235JR'#'S235JR'
    raw_data = pd.read_csv(path_data)
    #n = int(len(raw_data)/6)
    #raw_data = raw_data.iloc[n:2*n].reset_index(drop=True)
    tool_diameter = 10
    part_position = [-35.64, 175.0, 354.94]
    part_dimension = [75.0, 75.0*2, 50.0, 0.1]
    path_material_constant = 'material_constant.csv'
    material_setting = machine_state.read_machine_state(path_material_constant)
    print(material_setting.Material)

    if material == 'S235JR':
        material_choose = 0
    elif material == 'AL_2007_T4':
        material_choose = 1
    else:
        material_choose = int(input("Material wählen:"))

    new_machine_state, tool, part = machine_state.set_machine_state(material_setting, material_choose,
                                                                    tool_diameter, part_position,
                                                                    part_dimension)
    true_curr, process_value = split_curr_data(raw_data)
    a_p_change = False
    target_frequency = 50
    circle_part = [False]
    data_df = monitoring.state_monitoring(new_machine_state, a_p_change, tool,
                                          part, process_value, true_curr,
                                          target_frequency, circle_part, False)

    plt.plot(data_df['f_x_sim'])
    plt.show()
    data_df.to_csv('data.csv')