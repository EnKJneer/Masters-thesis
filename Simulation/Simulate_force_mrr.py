import matplotlib.pyplot as plt
import pandas as pd

import monitoring
import machine_state
import matplotlib
matplotlib.use('TkAgg')  # oder ein anderes interaktives Backend

def split_curr_data(data: pd.DataFrame) -> tuple:
    curr_column = ['curr_x', 'curr_y', 'curr_z', 'curr_sp']

    # Erste DataFrame mit den Spalten curr_x, curr_y, curr_z, curr_sp
    true_curr = data[curr_column]

    # Zweite DataFrame mit den restlichen Spalten
    process_value = data.drop(columns=curr_column)

    return true_curr, process_value

if __name__ == '__main__':
    path_data = '..\\DataSetsV3/DataMerged/AL_2007_T4_Plate_Normal.csv'

    material = 'AL_2007_T4'

    raw_data = pd.read_csv(path_data)
    n = int(len(raw_data)/6)
    raw_data = raw_data.iloc[n:2*n].reset_index(drop=True)

    tool_diameter = 10 # ToDo: Prüfen ob das stimmt und nötig ist. -> Modularer gestalten
    part_position = [-35.64, 175.0, 354.94] # ToDo: Prüfen ob das stimmt und nötig ist. -> Modularer gestalten
    part_dimension= [75.0, 75.0*2, 50.0, 0.2]   # ToDo: Prüfen ob das stimmt und nötig ist. -> Modularer gestalten

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
    a_p_change = False # Keine Ahnung was das macht. ap ist die Eingriffstiefe. -> Gibt an, ob sich die Eingriffstiefe ändert
    target_frequency = 50
    circle_part = [False] # wird hoffentlich nicht gebraucht hat mit dem MRR zutun
    data_df = monitoring.state_monitoring(new_machine_state, a_p_change, tool,
                                                                part, process_value, true_curr,
                                                                target_frequency,
                                                                circle_part, False)

    plt.plot(data_df['f_x_sim'])
    plt.show()
    data_df.to_csv('data.csv')