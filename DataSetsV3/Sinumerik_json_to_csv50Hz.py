import os
from collections import Counter

import numpy as np
import pandas as pd
import json

from scipy.interpolate import make_interp_spline
from scipy.signal import firwin, filtfilt


# file to read json-files in a folder and export data as csv-file
def extract_data_commands(data, names, commands, axis_translations, command_translations, current_fs = 500):
    extracted_data = {}
    for command in commands:
        if command != 'CYCLE':
            name = command_translations[command]
            for axis in axis_translations:
                key = f"{name}_{axis}"
                index = names.index(f"{command}|{axis_translations[axis]}")
                extracted_data[key] = data[index]
        else:
            index = names.index('CYCLE')
            extracted_data['time'] = (np.array(data[index]) - np.array(data[index])[0]) * 1/current_fs
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

def apply_fir_filter(df, cutoff_freq, fs):
    """Apply FIR filter to the data."""
    numtaps = 101
    fir_coeff = firwin(numtaps, cutoff_freq, fs=fs)
    df_filtered = df.copy()
    df_filtered = df_filtered.apply(lambda x: filtfilt(fir_coeff, 1.0, x))
    return df_filtered

def apply_moving_average_filter(df, window_size):
    """Apply a moving average filter to the data and fill NaN values forward and backward."""
    df_filtered = df.copy()
    df_filtered = df_filtered.rolling(window=window_size, center=True).mean()
    df_filtered = df_filtered.ffill().bfill()
    return df_filtered

fs_target = 50
fs_current = 500
cutoff_freq = 24

path_list = ['RawData']
path_target = 'DataSinumerik_50Hz'

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

            commands = ['CYCLE', 'ENC_POS', 'CURRENT', 'DES_POS', 'CMD_SPEED', 'CONT_DEV']
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

            new_df = calculate_derivatives(new_df, fs_current)

            # Differenz zwischen aufeinanderfolgenden Zeitstempeln berechnen
            new_df['time_diff'] = new_df['time'].diff().round(6)

            # Häufigste Differenz bestimmen
            most_common_diff = Counter(new_df['time_diff'].dropna()).most_common(1)[0][0]

            # Bereiche identifizieren, in denen die Differenz nicht der häufigsten Differenz entspricht
            non_constant_areas = new_df[new_df['time_diff'] != most_common_diff]
            non_constant_areas = non_constant_areas.dropna()
            print(f'Zeitsprünge: {non_constant_areas.shape[0]}')

            assert non_constant_areas.shape[0] == 0, 'Zeitsprung erkannt'

            downsampling_factor = int(round(fs_current / fs_target))
            print(f'Down sampling factor: {downsampling_factor}')
            print(f'New sampling frequency: {fs_current / downsampling_factor}')

            if downsampling_factor > 1:
                df_filtered = apply_fir_filter(new_df, cutoff_freq, fs_target)
                df_filtered = apply_moving_average_filter(df_filtered, int(downsampling_factor / 2))
                df_resampled = df_filtered.iloc[::downsampling_factor].reset_index(drop=True)
                print(new_df['time'].iloc[-1])
                print(df_resampled['time'].iloc[-1])
            else:
                df_resampled = new_df

            name = file.replace('.json', '.csv') #_SINUMERIK
            file_path = os.path.join(path_target, name)
            print(f'Saving {file_path}')

            # Save Data as CSV
            df_resampled.to_csv(file_path, index=False)