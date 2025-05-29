import os
import pandas as pd
import json

# file to read json-files in an foulder and export data as csv-file

path_list = ['RawData/AL_2007_T4/Training/AL_2007_T4_Plate_SF']

path_target = 'AdditionalData'
for path in path_list:

    machine = 'CMX'  # CMX, DMU
    Data_file = []  # Initialize the data list
    names = []  # Initialize the name List

    folder = os.listdir(path)

    # read all json files in the selected folder an add the Data to Data list
    for file in folder:
        if file.endswith('.json'):
            with open(path + '/' + file) as f:
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


            def extract_data_commands(data, names, commands, axis_translations):
                extracted_data = {}
                for command in commands:
                    for axis in axis_translations:
                        key = f"{command}_{axis}"
                        index = names.index(f"{command}|{axis_translations[axis]}")
                        extracted_data[key] = data[index]
                return extracted_data


            commands = ['DES_POS', 'ENC_POS', 'ENC1_POS', 'ENC2_POS', 'CURRENT', 'CTRL_DIFF', 'CTRL_DIFF2', 'CONT_DEV', 'CMD_SPEED']
            axis_translations = {'X': '1', 'Y': '2', 'Z': '3', 'SP': '6'}

            raw_data = extract_data_commands(Data, names, commands, axis_translations)

            new_df = pd.DataFrame(data=raw_data)

            # Save Data as CSV
            new_df.to_csv(path_target + '/' + path.split('/')[-1] + '.csv', index=False)