import os

import numpy as np
import pandas as pd
import json

# file to read json-files in a folder and export data as csv-file
def extract_data_commands(data, names, commands, axis_translations, command_translations):
    extracted_data = {}
    for command in commands:
        name = command_translations[command]
        for axis in axis_translations:
            key = f"{name}_{axis}"
            index = names.index(f"{command}|{axis_translations[axis]}")
            extracted_data[key] = data[index]
    return extracted_data


def calculate_derivatives(data, sampling_rate=500):
    """
    Berechnet Geschwindigkeiten und Beschleunigungen aus Positionsdaten
    """
    # Alle Positionsspalten finden
    pos_cols = [col for col in data.columns if col.startswith('pos_')]

    print(f"Berechne Ableitungen für Spalten: {pos_cols}")

    dt = 1.0 / sampling_rate  # Zeitschritt

    for pos_col in pos_cols:
        # Achsenbezeichnung extrahieren (z.B. 'X', 'Y', 'Z', 'SP')
        axis = pos_col.replace('pos_', '')

        # Geschwindigkeit berechnen (erste Ableitung)
        v_col = f'v_{axis}'
        data[v_col] = np.gradient(data[pos_col], dt)

        # Beschleunigung berechnen (zweite Ableitung)
        a_col = f'a_{axis}'
        data[a_col] = np.gradient(data[v_col], dt)

    return data

path_list = ['RawData']
path_target = 'RawDataSinumerik'

for path in path_list:
    machine = 'CMX'  # CMX, DMU
    folder = os.listdir(path)
    files = ['S235JR_Plate_Normal_SINUMERIK.csv']

    # Read all JSON files in the selected folder and process the data
    for file in folder:
        if file.endswith('.json'):
            print(f'Processing {file}')

            # Initialize the data and names lists for each file
            Data_file = []
            names = []

            with open(os.path.join(path, file)) as f:
                # Load data
                raw_data = json.load(f)

                # Get the names of the time series
                for i in raw_data['Header']['SignalListHFData']:
                    names += [list(i.values())[3]]

                # Get the data of the time series
                for data in raw_data['Payload']:
                    if 'HFData' in data.keys():
                        for row in data['HFData']:
                            Data_file += [row]

            # Transpose data
            Data = [[Data_file[s][i] for s in range(len(Data_file))] for i in range(len(Data_file[0]))]

            commands = ['ENC_POS', 'CURRENT', 'DES_POS', 'CMD_SPEED', 'CONT_DEV']
            axis_translations = {'x': '1', 'y': '2', 'z': '3', 'sp': '6'}

            # Spalten der SINUMERIK Daten umbenennen
            column_mapping = {
                'ENC_POS': 'pos',
                'CURRENT': 'curr',
                'DES_POS': 'cmd_pos',
                'CMD_SPEED': 'cmd_v',
                'CONT_DEV': 'cont_dev',
            }

            raw_data = extract_data_commands(Data, names, commands, axis_translations, column_mapping)

            new_df = pd.DataFrame(data=raw_data)

            new_df = calculate_derivatives(new_df, 500)

            name = file.replace('.json', '.csv') #_SINUMERIK
            file_path = os.path.join(path_target, name)
            print(f'Saving {file_path}')

            # Save Data as CSV
            new_df.to_csv(file_path, index=False)