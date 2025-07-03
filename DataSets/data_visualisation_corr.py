import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Pfad zu den Dateien
path_data = 'DataFiltered'

# Liste der Dateien
files = ['AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv',
         'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']

#files = ['AL_2007_T4_Plate_Normal_3.csv']
# Anzahl der letzten Datenpunkte, die ausgeschlossen werden sollen
n = 25

def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-(x + b))) + c

# Iteriere über die Dateien
for file in files:
    if '_3' in file:
        # Lade die Daten
        data = pd.read_csv(os.path.join(path_data, file))
        print(f"Columns in {file}: {data.columns}")
        print(f"Shape of data in {file}: {data.shape}")

        # Entferne die letzten n Datenpunkte
        data = data.iloc[:-n]

        # Berechne die Komponenten der Materialentfernung
        f_x = data['f_x_sim']
        f_y = data['f_y_sim']
        f_z = data['f_z_sim']
        v_x = data['v_x']
        v_y = data['v_y']
        v_z = data['v_z']
        data['mrr_x'] = data['materialremoved_sim'] * (np.abs(v_x) / (np.abs(v_x) + np.abs(v_y) + 1e-10))
        data['mrr_y'] = data['materialremoved_sim'] * (np.abs(v_y) / (np.abs(v_x) + np.abs(v_y) + 1e-10))
        data['mrr_z'] = data['materialremoved_sim'] * (np.abs(v_z) / (np.abs(v_x) + np.abs(v_y) + 1e-10))

        # Definiere die Achsen
        axes = ['x']

        # Berechne die Korrelationsmatrix für curr_x
        corr_matrix = data[['curr_x', 'curr_y', 'pos_x', 'v_x', 'f_x_sim', 'materialremoved_sim', 'f_y_sim', 'f_z_sim', 'f_sp_sim']].corr()

        # Plotte die Korrelationsmatrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'{file}: Korrelationsmatrix für curr_x und andere Komponenten')
        plt.show()

        # Iteriere über die Achsen
        for axis in axes:
            y = data[f'curr_{axis}']
            data['t1'] = np.clip(data[f'a_{axis}'] * data[f'v_{axis}'], -25, 25)
            data['t2'] = data[f'v_{axis}'] ** 2 * np.sign(data[f'v_{axis}'])
            initial_params_sigmoid = [max(y) - min(y), np.median(data[f'v_{axis}']), min(y)]
            data['t2_s'] = sigmoid(data[f'v_{axis}'], initial_params_sigmoid[0], initial_params_sigmoid[1], initial_params_sigmoid[2])
            data[f't3_{axis}'] = data[f'f_{axis}_sim'] * data[f'mrr_{axis}']
            data['t3'] = data[f'f_{axis}_sim'] * data['materialremoved_sim']

            # Berechne die Korrelationsmatrix für curr_x
            corr_matrix = data[[f'curr_{axis}', 't1', 't2', 't3', f't3_{axis}', f'v_{axis}', f'f_{axis}_sim', 't2_s']].corr()

            # Plotte die Korrelationsmatrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f'{file}: Korrelationsmatrix für curr_{axis} und andere Komponenten')
            plt.show()

            # Plotte die Zusammenhänge für jede Achse
            plt.figure(figsize=(18, 12))

            # Streudiagramm von v vs. curr
            plt.subplot(2, 2, 1)
            plt.scatter(data[f'v_{axis}'], data[f'curr_{axis}'], alpha=0.5)
            plt.xlabel(f'v_{axis}')
            plt.ylabel(f'curr_{axis}')
            plt.title(f'{file}: v_{axis} vs. curr_{axis}')

            # Streudiagramm von f_sim vs. curr
            plt.subplot(2, 2, 2)
            plt.scatter(data[f'f_{axis}_sim'], data[f'curr_{axis}'], alpha=0.5)
            plt.xlabel(f'f_{axis}_sim')
            plt.ylabel(f'curr_{axis}')
            plt.title(f'{file}: f_{axis}_sim vs. curr_{axis}')

            """            
            if axis == 'x' or axis == 'y':
                # Streudiagramm von mrr vs. curr
                plt.subplot(2, 2, 3)
                plt.scatter(data[f'mrr_{axis}'], data[f'curr_{axis}'], alpha=0.5)
                plt.xlabel(f'mrr_{axis}')
                plt.ylabel(f'curr_{axis}')
                plt.title(f'{file}: mrr_{axis} vs. curr_{axis}')"""

            # Streudiagramm von mrr vs. curr
            plt.subplot(2, 2, 3)
            plt.scatter(data[f'pos_x'], data[f'curr_{axis}'], alpha=0.5)
            plt.xlabel(f'pos_x')
            plt.ylabel(f'curr_{axis}')
            plt.title(f'{file}: pos_x vs. curr_{axis}')

            # Streudiagramm von materialremoved_sim vs. curr
            plt.subplot(2, 2, 4)
            #plt.scatter(data['materialremoved_sim'], data[f'curr_{axis}'], alpha=0.5)
            #plt.xlabel('materialremoved_sim')
            #plt.ylabel(f'curr_{axis}')
            #plt.title(f'{file}: materialremoved_sim vs. curr_{axis}')
            if axis == 'x':
                t =  data[f'f_y_sim']
                plt.scatter(t, data[f'curr_{axis}'], alpha=0.5)
                plt.xlabel('f_y_sim')
                plt.ylabel(f'curr_{axis}')
                plt.title(f'{file}: f_y_sim vs. curr_{axis}')
            elif axis == 'y':
                t =  data[f'f_x_sim']
                plt.scatter(t, data[f'curr_{axis}'], alpha=0.5)
                plt.xlabel('f_x_sim')
                plt.ylabel(f'curr_{axis}')
                plt.title(f'{file}: f_x_sim vs. curr_{axis}')
            plt.tight_layout()
            plt.show()

            # Untersuchung Erd
            # Plotte die Zusammenhänge für jede Achse
            plt.figure(figsize=(18, 12))

            # Streudiagramm von v vs. curr
            plt.subplot(2, 2, 1)
            t1 = np.clip(data[f'a_{axis}'] * data[f'v_{axis}'], -25, 25)
            plt.scatter(t1, data[f'curr_{axis}'], alpha=0.5)
            plt.xlabel(f'Term 1')
            plt.ylabel(f'curr_{axis}')
            plt.title(f'{file}: a_{axis}*v_{axis} vs. curr_{axis}')

            # Streudiagramm von f_sim vs. curr
            plt.subplot(2, 2, 2)
            t2 = data[f'v_{axis}']**2 * np.sign(data[f'v_{axis}'])
            plt.scatter(data[f'f_{axis}_sim'], data[f'curr_{axis}'], alpha=0.5)
            plt.xlabel(f'Term 2')
            plt.ylabel(f'curr_{axis}')
            plt.title(f'{file}: v_{axis}**2 * sign(v_{axis}) vs. curr_{axis}')

            if axis == 'x' or axis == 'y':
                # Streudiagramm von mrr vs. curr
                plt.subplot(2, 2, 3)
                t3 = data[f'f_{axis}_sim'] * data[f'mrr_{axis}']
                plt.scatter(t3, data[f'curr_{axis}'], alpha=0.5)
                plt.xlabel(f'Term 3 x Komponente')
                plt.ylabel(f'curr_{axis}')
                plt.title(f'{file}: f_{axis}_sim * mrr_{axis} vs. curr_{axis}')

            # Streudiagramm von materialremoved_sim vs. curr
            plt.subplot(2, 2, 4)
            # plt.scatter(data['materialremoved_sim'], data[f'curr_{axis}'], alpha=0.5)
            # plt.xlabel('materialremoved_sim')
            # plt.ylabel(f'curr_{axis}')
            t3 = data[f'f_{axis}_sim'] * data['materialremoved_sim']
            plt.scatter(t3, data[f'curr_{axis}'], alpha=0.5)
            plt.xlabel(f'Term 3')
            plt.ylabel(f'curr_{axis}')
            plt.title(f'{file}: f_{axis}_sim * mrr vs. curr_{axis}')

            """# Zeitverlauf der Variablen für jede Achse
            plt.figure(figsize=(18, 12))

            # Zeitverlauf von v
            plt.subplot(2, 2, 1)
            plt.plot(data.index, data[f'v_{axis}'], label=f'v_{axis}', color='blue')
            plt.xlabel('Index')
            plt.ylabel(f'v_{axis}')
            plt.title(f'{file}: Zeitverlauf von v_{axis}')

            # Zeitverlauf von f_sim
            plt.subplot(2, 2, 2)
            plt.plot(data.index, data[f'f_{axis}_sim'], label=f'f_{axis}_sim', color='green')
            plt.xlabel('Index')
            plt.ylabel(f'f_{axis}_sim')
            plt.title(f'{file}: Zeitverlauf von f_{axis}_sim')

            # Zeitverlauf von mrr
            plt.subplot(2, 2, 3)
            plt.plot(data.index, data[f'mrr_{axis}'], label=f'mrr_{axis}', color='red')
            plt.xlabel('Index')
            plt.ylabel(f'mrr_{axis}')
            plt.title(f'{file}: Zeitverlauf von mrr_{axis}')

            # Zeitverlauf von curr
            plt.subplot(2, 2, 4)
            plt.plot(data.index, data[f'curr_{axis}'], label=f'curr_{axis}', color='purple')
            plt.xlabel('Index')
            plt.ylabel(f'curr_{axis}')
            plt.title(f'{file}: Zeitverlauf von curr_{axis}')

            plt.tight_layout()
            plt.show()
"""