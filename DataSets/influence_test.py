import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear(x, a, b):
    return a * x + b

def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-(x + b))) + c

def combined_model(x, a1, b1, a2, b2, c2):
    return linear(x[0], a1, b1) + sigmoid(x[1], a2, b2, c2)

def combined_model_linear(x, a1, a2, b2):
    return linear(x[0], a1, 0) + linear(x[1], a2, b2)

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def combined_model_y(x, a1, a2, b2, c2, a3, b3):
    #one = 1 if np.all(x[0] == 0) else 0
    return linear(x[0], a1, 0) + sigmoid(x[1], a2, b2, c2) + a3 * sigmoid(x[2], a3, b3, 0)

def combined_model_linear_3(x, a, b, c, d):
    x_1, x_2, x_3 = x
    return a * x_1 + b * x_2 + c * x_3 + d

def combined_model_linear_4(x, a, b, c, d, e):
    x_1, x_2, x_3, x_4 = x
    return a * x_1 + b * x_2 + c * x_3 + d * x_4 + e

path_data = 'DataFiltered'

files = os.listdir(path_data)
files = ['AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv',
         'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']

n = 25
mae_values = {file: [] for file in files}

for file in files:
    if '_3' in file:
        data = pd.read_csv(f'{path_data}/{file}')
        print(f"Columns in {file}: {data.columns}")
        print(f"Shape of data in {file}: {data.shape}")

        f_x = data['f_x_sim'].iloc[:-n].values
        f_y = data['f_y_sim'].iloc[:-n].values
        pos_x = data['pos_x'].iloc[:-n].values
        v_x = data['v_x'].iloc[:-n].values
        v_y = data['v_y'].iloc[:-n].values
        a_x = data['a_x'].iloc[:-n].values
        a_y = data['a_y'].iloc[:-n].values
        y = data['curr_x'].iloc[:-n].values
        mrr_x = data['materialremoved_sim'].iloc[:-n].values * np.abs(v_x) / (np.abs(v_x) + np.abs(v_y))
        mrr_x = np.nan_to_num(mrr_x)
        print(np.isnan(mrr_x).any())

        # Kombinierte Gleichung mit f_x_sim und sigmoidaler Gleichung mit v_x
        initial_params_combined = [1, 1, 1, 1, 1]
        params_combined, _ = curve_fit(combined_model, (f_x, v_x), y, p0=initial_params_combined)
        y_pred_combined = combined_model((f_x, v_x), *params_combined)
        print(f'Parameter Combined: {params_combined}')
        mse_combined = calculate_mae(y, y_pred_combined)

        initial_params_combined_lin_2 = [1, 1, 1, 1, 1]
        xdata = (f_x, a_x, v_x, pos_x)
        params_combined_lin_2, _ = curve_fit(combined_model_linear_4, xdata, y, p0=initial_params_combined_lin_2)
        y_pred_combined_lin_2 = combined_model_linear_4(xdata, *params_combined_lin_2)
        print(f'Parameter Combined 2: {params_combined_lin_2}')
        mse_combined_lin_2 = calculate_mae(y, y_pred_combined_lin_2)

        # MSE Werte speichern
        mae_values[file].extend([mse_combined, mse_combined_lin_2])

        # Ergebnisse ausgeben

        print(f"MSE für kombinierte Gleichung: {mse_combined:.2f}")
        print(f"MSE für kombinierte Gleichung 2: {mse_combined_lin_2:.2f}")

        # Plots für den zeitlichen Verlauf
        plt.figure(figsize=(12, 6))
        plt.plot(data.index[:-n], y, label='Original curr_x', color='blue')

        plt.xlabel('Index')
        plt.ylabel('curr_x')
        plt.title(f'Zeitlicher Verlauf von curr_x für {file}')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(data.index[:-n], y, label='Original curr_x', color='blue')
        plt.plot(data.index[:-n], y_pred_combined, label='Kombinierte Gleichung', linestyle='--')
        plt.plot(data.index[:-n], y_pred_combined_lin_2, label='Kombinierte Gleichung 2', linestyle='--')
        plt.xlabel('Index')
        plt.ylabel('curr_x')
        plt.title(f'Zeitlicher Verlauf von curr_x für {file}')
        plt.legend()
        plt.show()

        diff = y - y_pred_combined

        
# Balkendiagramm der MSE Werte
plt.figure(figsize=(12, 6))
labels = ['Kombiniert sigmoid', 'Linear 4']
x = np.arange(len(labels))
width = 0.2

for i, file in enumerate(files):
    bars = plt.bar(x + i * width, mae_values[file], width, label=file)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')

plt.xlabel('Modell')
plt.ylabel('MAE')
plt.title('MAE Vergleich der Modelle')
plt.xticks(x + width, labels)
plt.legend()
plt.show()
