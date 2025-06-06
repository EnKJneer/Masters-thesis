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
        v_x = data['v_x'].iloc[:-n].values
        v_y = data['v_y'].iloc[:-n].values
        y = data['curr_x'].iloc[:-n].values
        mrr_y = data['materialremoved_sim'].iloc[:-n].values * (v_y) / (np.abs(v_x) + np.abs(v_y))
        mrr_y = np.nan_to_num(mrr_y)
        print(np.isnan(mrr_y).any())
        # Lineare Gleichung mit f_x_sim
        params_linear_fx, _ = curve_fit(linear, f_x, y)
        y_pred_linear_fx = linear(f_x, *params_linear_fx)
        mse_linear_fx = calculate_mae(y, y_pred_linear_fx)

        # Lineare Gleichung mit v_x
        params_linear_vx, _ = curve_fit(linear, v_y, y)
        y_pred_linear_vx = linear(v_y, *params_linear_vx)
        mse_linear_vx = calculate_mae(y, y_pred_linear_vx)

        # Lineare Gleichung mit v_x²*sign(v_x)
        x = v_x**2 * np.sign(v_x)
        params_linear_vx_s, _ = curve_fit(linear, x, y)
        y_pred_linear_vx_s = linear(x, *params_linear_vx_s)
        print(f'Parameter quadratic v_x: {params_linear_vx_s}')
        mse_linear_vx_s = calculate_mae(y, y_pred_linear_vx_s)

        # Sigmoidale Gleichung mit v_x
        initial_params_sigmoid = [max(y) - min(y), np.median(v_x), min(y)]
        params_sigmoid_vx, _ = curve_fit(sigmoid, v_x, y, p0=initial_params_sigmoid)
        y_pred_sigmoid_vx = sigmoid(v_x, *params_sigmoid_vx)
        mse_sigmoid_vx = calculate_mae(y, y_pred_sigmoid_vx)

        # Kombinierte Gleichung mit f_x_sim und sigmoidaler Gleichung mit v_x
        initial_params_combined = [1, 1, 1, 1, 1]
        params_combined, _ = curve_fit(combined_model, (f_x, v_x), y, p0=initial_params_combined)
        y_pred_combined = combined_model((f_x, v_x), *params_combined)
        print(f'Parameter Combined: {params_combined}')
        mse_combined = calculate_mae(y, y_pred_combined)

        # Kombinierte Gleichung mit f_x_sim und sigmoidaler linear mit v_x
        initial_params_combined_lin = [1, 1, 1]
        v = v_x**2 * np.sign(v_x)
        params_combined_lin, _ = curve_fit(combined_model_linear, (f_x, v), y, p0=initial_params_combined_lin)
        y_pred_combined_lin = combined_model_linear((f_x, v), *params_combined_lin)
        print(f'Parameter Combined linear: {params_combined_lin}')
        mse_combined_lin = calculate_mae(y, y_pred_combined_lin)

        initial_params_combined_lin_2 = [1, 1, 1, 1, 1, 1]
        params_combined_lin_2, _ = curve_fit(combined_model_y, (f_x, v_x, f_y), y, p0=initial_params_combined_lin_2)
        y_pred_combined_lin_2 = combined_model_y((f_x, v_x, f_y), *params_combined_lin_2)
        print(f'Parameter Combined 2: {params_combined_lin_2}')
        mse_combined_lin_2 = calculate_mae(y, y_pred_combined_lin_2)

        # MSE Werte speichern
        mae_values[file].extend([mse_linear_fx, mse_linear_vx, mse_linear_vx_s, mse_sigmoid_vx,
                                 mse_combined, mse_combined_lin, mse_combined_lin_2])

        # Ergebnisse ausgeben
        print(f"MSE für lineare Gleichung mit f_x_sim: {mse_linear_fx:.2f}")
        print(f"MSE für lineare Gleichung mit v_x: {mse_linear_vx:.2f}")
        print(f"MSE für lineare Gleichung mit v_x²*sign(v_x): {mse_linear_vx_s:.2f}")
        print(f"MSE für sigmoidale Gleichung mit v_x: {mse_sigmoid_vx:.2f}")
        print(f"MSE für kombinierte Gleichung: {mse_combined:.2f}")
        print(f"MSE für kombinierte Gleichung Linear: {mse_combined_lin:.2f}")
        print(f"MSE für kombinierte Gleichung 2: {mse_combined_lin_2:.2f}")

        # Plots für den zeitlichen Verlauf
        plt.figure(figsize=(12, 6))
        plt.plot(data.index[:-n], y, label='Original curr_x', color='blue')
        plt.plot(data.index[:-n], y_pred_linear_fx, label='Lineare Gleichung mit f_x_sim', linestyle='--')
        plt.plot(data.index[:-n], y_pred_linear_vx, label='Lineare Gleichung mit v_x', linestyle='--')
        plt.plot(data.index[:-n], y_pred_linear_vx, label='Lineare Gleichung mit v_x²*sign(v_x)', linestyle='--')
        plt.plot(data.index[:-n], y_pred_sigmoid_vx, label='Sigmoidale Gleichung mit v_x', linestyle='--')
        plt.xlabel('Index')
        plt.ylabel('curr_x')
        plt.title(f'Zeitlicher Verlauf von curr_x für {file}')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(data.index[:-n], y, label='Original curr_x', color='blue')
        plt.plot(data.index[:-n], y_pred_combined, label='Kombinierte Gleichung', linestyle='--')
        plt.plot(data.index[:-n], y_pred_combined_lin, label='Kombinierte Gleichung linear', linestyle='--')
        plt.plot(data.index[:-n], y_pred_combined_lin_2, label='Kombinierte Gleichung 2', linestyle='--')
        plt.xlabel('Index')
        plt.ylabel('curr_x')
        plt.title(f'Zeitlicher Verlauf von curr_x für {file}')
        plt.legend()
        plt.show()

        diff = y - y_pred_combined

        
# Balkendiagramm der MSE Werte
plt.figure(figsize=(12, 6))
labels = ['Linear f_x_sim', 'Linear v_x', 'Linear v_x²*sgn(v_x)', 'Sigmoid v_x',
          'Kombiniert', 'Kombiniert linear', 'Kombiniert 2']
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
