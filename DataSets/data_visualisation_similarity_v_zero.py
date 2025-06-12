import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def sign_hold(v_x):
    signs = np.sign(v_x)
    nonzero_idx = np.nonzero(signs)[0]

    if len(nonzero_idx) == 0:
        return signs

    indices = np.searchsorted(nonzero_idx, np.arange(len(signs)), side='right') - 1
    indices = np.clip(indices, 0, len(nonzero_idx) - 1)

    result = signs.copy()
    zero_mask = (signs == 0)
    valid_replacement = indices >= 0

    result[zero_mask & valid_replacement] = signs[nonzero_idx[indices[zero_mask & valid_replacement]]]

    return result

# Pfad zu den Dateien
path_data = 'DataFiltered'

# Liste der Dateien
files = ['AL_2007_T4_Plate_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']
n = 25

def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-(x + b))) + c

# Iteriere über die Dateien
for file in files:
    data = pd.read_csv(os.path.join(path_data, file))
    print(f"Columns in {file}: {data.columns}")
    print(f"Shape of data in {file}: {data.shape}")

    data = data.iloc[:-n]

    v_x = data['v_x']
    v_y = data['v_y']
    v_z = data['v_z']
    data['mrr_x'] = data['materialremoved_sim'] * (np.abs(v_x) / (np.abs(v_x) + np.abs(v_y) + 1e-10))
    data['mrr_y'] = data['materialremoved_sim'] * (np.abs(v_y) / (np.abs(v_x) + np.abs(v_y) + 1e-10))
    data['mrr_z'] = data['materialremoved_sim'] * (np.abs(v_z) / (np.abs(v_x) + np.abs(v_y) + 1e-10))
    data['v_x_hold'] = -sign_hold(v_x)
    data['v_y_hold'] = -sign_hold(v_y)
    axes = ['x', 'y']

    corr_matrix = data[['curr_x', 'curr_y', 'v_x', 'v_y', 'f_x_sim', 'f_y_sim', 'materialremoved_sim', 'curr_z', 'curr_sp']].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'{file}: Korrelationsmatrix für curr_x und andere Komponenten')
    plt.show()

    for axis in axes:
        eps = 0.01
        data_filtered = data.where(np.abs(data[f'v_{axis}']) < eps)
        data_filtered = data_filtered.dropna()
        data_filtered = data_filtered.set_axis(range(len(data_filtered)))
        y = data_filtered[f'curr_{axis}']
        data_filtered['t1'] = np.clip(data_filtered[f'a_{axis}'] * data_filtered[f'v_{axis}'], -25, 25)
        data_filtered['t2'] = data_filtered[f'v_{axis}'] ** 2 * np.sign(data_filtered[f'v_{axis}'])
        initial_params_sigmoid = [max(y) - min(y), np.median(data_filtered[f'v_{axis}']), min(y)]
        data_filtered['t2_s'] = sigmoid(data_filtered[f'v_{axis}'], initial_params_sigmoid[0], initial_params_sigmoid[1], initial_params_sigmoid[2])
        data_filtered[f't3_{axis}'] = data_filtered[f'f_{axis}_sim'] * data_filtered[f'mrr_{axis}']
        data_filtered['t3'] = data_filtered[f'f_{axis}_sim'] * data_filtered['materialremoved_sim']

        plt.figure(figsize=(18, 12))

        plt.subplot(2, 2, 1)
        if axis == 'x':
            plt.scatter(data_filtered[f'f_y_sim'], data_filtered[f'curr_{axis}'], alpha=0.5)
            plt.xlabel(f'f_y_sim')
            plt.ylabel(f'curr_{axis}')
            plt.title(f'{file}: f_y_sim vs. curr_{axis}')
        elif axis == 'y':
            plt.scatter(data_filtered[f'f_x_sim'], data_filtered[f'curr_{axis}'], alpha=0.5)
            plt.xlabel(f'f_x_sim')
            plt.ylabel(f'curr_{axis}')
            plt.title(f'{file}: f_x_sim vs. curr_{axis}')

        plt.subplot(2, 2, 2)
        plt.scatter(data_filtered[f'f_{axis}_sim'], data_filtered[f'curr_{axis}'], alpha=0.5)
        plt.xlabel(f'f_{axis}_sim')
        plt.ylabel(f'curr_{axis}')
        plt.title(f'{file}: f_{axis}_sim vs. curr_{axis}')

        plt.subplot(2, 2, 3)
        plt.scatter(data_filtered[f'materialremoved_sim'], data_filtered[f'curr_{axis}'], alpha=0.5)
        plt.xlabel(f'materialremoved_sim')
        plt.ylabel(f'curr_{axis}')
        plt.title(f'{file}: materialremoved_sim vs. curr_{axis}')

        plt.subplot(2, 2, 4)
        t3 = data_filtered[f'f_{axis}_sim'] * data_filtered['materialremoved_sim']
        plt.scatter(t3, data_filtered[f'curr_{axis}'], alpha=0.5)
        plt.xlabel(f'Term 3')
        plt.ylabel(f'curr_{axis}')
        plt.title(f'{file}: f_{axis}_sim * mrr vs. curr_{axis}')

        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(18, 12))

        if axis == 'x':
            axis_2 = 'y'
        elif axis == 'y':
            axis_2 = 'x'

        plt.subplot(2, 3, 1)
        plt.plot(data_filtered.index, data_filtered[f'v_{axis_2}'], label=f'v_{axis_2}', color='blue')
        plt.xlabel('Index')
        plt.ylabel(f'v_{axis_2}')
        plt.title(f'{file}: Zeitverlauf von v_{axis_2}')
        plt.twinx()
        plt.plot(data_filtered.index, data_filtered[f'curr_{axis}'], label=f'curr_{axis}', color='purple', linestyle='--')
        plt.ylabel(f'curr_{axis}')

        plt.subplot(2, 3, 2)
        plt.plot(data_filtered.index, data_filtered[f'f_{axis}_sim'], label=f'f_{axis}_sim', color='green')
        plt.xlabel('Index')
        plt.ylabel(f'f_{axis}_sim')
        plt.title(f'{file}: Zeitverlauf von f_{axis}_sim')
        plt.twinx()
        plt.plot(data_filtered.index, data_filtered[f'curr_{axis}'], label=f'curr_{axis}', color='purple', linestyle='--')
        plt.ylabel(f'curr_{axis}')

        plt.subplot(2, 3, 3)
        plt.plot(data_filtered.index, data_filtered[f'f_{axis_2}_sim'], label=f'f_{axis_2}_sim', color='green')
        plt.xlabel('Index')
        plt.ylabel(f'f_{axis_2}_sim')
        plt.title(f'{file}: Zeitverlauf von f_{axis_2}_sim')
        plt.twinx()
        plt.plot(data_filtered.index, data_filtered[f'curr_{axis}'], label=f'curr_{axis}', color='purple', linestyle='--')
        plt.ylabel(f'curr_{axis}')

        plt.subplot(2, 3, 4)
        plt.plot(data_filtered.index, data_filtered[f'materialremoved_sim'], label=f'materialremoved_sim', color='red')
        plt.xlabel('Index')
        plt.ylabel(f'materialremoved_sim')
        plt.title(f'{file}: Zeitverlauf von materialremoved_sim')
        plt.twinx()
        plt.plot(data_filtered.index, data_filtered[f'curr_{axis}'], label=f'curr_{axis}', color='purple', linestyle='--')
        plt.ylabel(f'curr_{axis}')

        plt.subplot(2, 3, 5)
        plt.plot(data_filtered.index, data_filtered[f'curr_{axis_2}'], label=f'curr_{axis_2}', color='orange')
        plt.xlabel('Index')
        plt.ylabel(f'curr_{axis_2}')
        plt.title(f'{file}: Zeitverlauf von curr_{axis_2}')
        plt.twinx()
        plt.plot(data_filtered.index, data_filtered[f'curr_{axis}'], label=f'curr_{axis}', color='purple', linestyle='--')
        plt.ylabel(f'curr_{axis}')

        plt.subplot(2, 3, 6)
        plt.plot(data_filtered.index, data_filtered[f'v_{axis}_hold'], label=f'v_{axis}_hold', color='brown')
        plt.xlabel('Index')
        plt.ylabel(f'v_{axis}_hold')
        plt.title(f'{file}: Zeitverlauf von v_{axis}_hold')
        plt.twinx()
        plt.plot(data_filtered.index, data_filtered[f'curr_{axis}'], label=f'curr_{axis}', color='purple', linestyle='--')
        plt.ylabel(f'curr_{axis}')

        plt.tight_layout()
        plt.show()
