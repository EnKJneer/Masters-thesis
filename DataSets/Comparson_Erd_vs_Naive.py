import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Funktion zur Berechnung des Mean Absolute Error (MAE)
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Kombiniertes Modell für die Kurvenanpassung
def combined_model_linear(x, a, b, c):
    x_1, x_2 = x
    return a * x_1 + b * x_2 + c

def combined_model_linear_3(x, a, b, c, d):
    x_1, x_2, x_3 = x
    return a * x_1 + b * x_2 + c * x_3 + d

def model_erd(x, a, b, c, d, e):
    x_1, x_2, x_3, x_4 = x
    return a * x_1 + b * x_2 + c * x_3 + d * x_4 + e


def combined_model_linear_3_dual(x,
                                 a_pos, b_pos, c_pos, d_pos,
                                 a_neg, b_neg, c_neg, d_neg):
    x_1, x_2, x_3 = x

    eps = 0.1
    mask_pos = np.abs(x_3) >= eps
    mask_neg = np.abs(x_3) < eps

    # Initialisierung
    y_pred = np.zeros_like(x_1)

    # Vorhersage für alle Punkte mit aktivem Signal
    y_pred[mask_pos] = a_pos * x_1[mask_pos] + b_pos * x_2[mask_pos] + c_pos * x_3[mask_pos] + d_pos

    # Sample-and-Hold für ruhige Phasen
    last_value = 0.0
    for i in range(len(x_1)):
        if mask_pos[i]:
            # Aktiver Bereich: berechne regulär und speichere Wert
            last_value = 0 #y_pred[i]
        elif mask_neg[i]:
            # Ruhiger Bereich: normaler Vorhersagewert PLUS letztes aktives Signal
            y_pred[i] = (
                a_pos * x_1[i] + a_pos * x_2[i] + c_neg * last_value + d_neg
            )
        # Optional: sonst nichts tun (z. B. falls du später weitere Masken nutzt)

    return y_pred

# Pfad zu den Dateien
path_data = 'DataFiltered'

# Liste der Dateien
files = ['AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv',
         'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']



# Liste zur Speicherung der MAE-Werte für alle Dateien
mae_values = []

# Iteriere über die Dateien
for file in files:
    if '_3' in file:

        # Lade die Daten
        data = pd.read_csv(os.path.join(path_data, file))
        # Anzahl der letzten Datenpunkte, die ausgeschlossen werden sollen
        n = 25
        # Berechne die Komponenten der Materialentfernung
        data['mrr_x'] = data['materialremoved_sim'] * (np.abs(data['v_x']) / (np.abs(data['v_x']) + np.abs(data['v_y']) + 1e-10))
        data['mrr_y'] = data['materialremoved_sim'] * (np.abs(data['v_y']) / (np.abs(data['v_x']) + np.abs(data['v_y']) + 1e-10))

        f_x = data['f_x_sim'].iloc[:-n].values
        f_y = data['f_y_sim'].iloc[:-n].values
        v_x = data['v_x'].iloc[:-n].values
        v_y = data['v_y'].iloc[:-n].values
        a_x = data['a_x'].iloc[:-n].values
        a_y = data['a_y'].iloc[:-n].values
        y = data['curr_x'].iloc[:-n].values
        mrr_x = data['mrr_x'].iloc[:-n].values
        mrr_y = data['mrr_y'].iloc[:-n].values

        axis = 'x'
        data[f't1_{axis}'] = data[f'a_{axis}'] * data[f'v_{axis}']
        data[f't2_{axis}'] = data[f'v_{axis}'] ** 2 * np.sign(data[f'v_{axis}'])
        data[f't2_{axis}_s'] = data[f'v_{axis}'] ** 2 * np.sign(data[f'v_{axis}'])
        data[f't3_{axis}'] = data[f'f_{axis}_sim'] * data['materialremoved_sim']
        t1 = data[f't1_{axis}'].iloc[:-n].values
        t2 = data[f't2_{axis}'].iloc[:-n].values
        t3 = data[f't3_{axis}'].iloc[:-n].values

        # Kombinierte Gleichung mit t3 und t2
        initial_params_combined_lin = [1, 1, 1, 1, 1]
        params_combined_lin, _ = curve_fit(model_erd, xdata=(t1, t2, t3, np.sign(v_x)), ydata=y, p0=initial_params_combined_lin)
        y_pred_combined_lin = model_erd((t1, t2, t3, np.sign(v_x)), *params_combined_lin)
        mae_combined_lin = calculate_mae(y, y_pred_combined_lin)
        mae_values.append(mae_combined_lin)
        print(f'Model Erd: {params_combined_lin}')

        # Plotte den zeitlichen Verlauf des Stroms, der Vorhersage und des Fehlers
        error = y - y_pred_combined_lin
        plt.figure(figsize=(12, 6))
        plt.plot(data.index[:-n], y, label='curr_x', color='blue')
        plt.plot(data.index[:-n], y_pred_combined_lin, label='Predicted curr_x', color='green')
        plt.plot(data.index[:-n], error, label='loss Erd', color='red')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Zeitlicher Verlauf von Strom, Vorhersage und Fehler für {file}')
        plt.legend()
        plt.show()

        # Kombinierte Gleichung mit t3 und sign(v_x)
        initial_params_combined_lin = [1, 1, 1, 1]
        params_combined_lin, _ = curve_fit(combined_model_linear_3, (a_x, f_x, np.sign(v_x)), y, p0=initial_params_combined_lin)
        y_pred_combined_lin = combined_model_linear_3((a_x, f_x, np.sign(v_x)), *params_combined_lin)
        mae_combined_lin = calculate_mae(y, y_pred_combined_lin)
        mae_values.append(mae_combined_lin)
        print(f'Model Naive: {params_combined_lin}')

        # Plotte den zeitlichen Verlauf des Stroms, der Vorhersage und des Fehlers
        error = y - y_pred_combined_lin
        plt.figure(figsize=(12, 6))
        plt.plot(data.index[:-n], y, label='curr_x', color='blue')
        plt.plot(data.index[:-n], y_pred_combined_lin, label='Predicted curr_x', color='green')
        plt.plot(data.index[:-n], error, label='loss Naive', color='red')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Zeitlicher Verlauf von Strom, Vorhersage und Fehler für {file}')
        plt.legend()
        plt.show()

        # Kombinierte Gleichung mit t3 und sign(v_x) und Hold
        initial_params_combined_lin_with_hold = [1, 1, 1, 1, 1, 1, 1, 1]
        params_combined_lin_with_hold, _ = curve_fit(
            combined_model_linear_3_dual,
            xdata=(a_x, f_x, np.sign(v_x)),
            ydata=y,
            p0=initial_params_combined_lin_with_hold
        )
        y_pred_combined_lin_with_hold = combined_model_linear_3_dual(
            (a_x, f_x, np.sign(v_x)),
            *params_combined_lin_with_hold
        )
        mae_combined_lin_with_hold = calculate_mae(y, y_pred_combined_lin_with_hold)
        mae_values.append(mae_combined_lin_with_hold)
        print(f'Model with Hold: {params_combined_lin_with_hold}')

        # Plotte den zeitlichen Verlauf des Stroms, der Vorhersage und des Fehlers
        error_with_hold = y - y_pred_combined_lin_with_hold
        plt.figure(figsize=(12, 6))
        plt.plot(data.index[:-n], y, label='curr_x', color='blue')
        plt.plot(data.index[:-n], y_pred_combined_lin_with_hold, label='Predicted curr_x with hold', color='purple')
        plt.plot(data.index[:-n], error_with_hold, label='loss with hold', color='orange')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Zeitlicher Verlauf von Strom, Vorhersage und Fehler für {file} mit Hold')
        plt.legend()
        plt.show()

plt.figure(figsize=(15, 8))

# Definiere Farben für jedes Modell
colors = {
    'Erd': 'blue',
    'Naive': 'green',
    'Hold': 'purple'
}

# Erstelle eine Liste von Farben basierend auf den Modellen
bar_colors = [colors['Erd'], colors['Naive'], colors['Hold']] * len(files)

bars = plt.bar(range(len(mae_values)), mae_values, color=bar_colors)
plt.xlabel('Modell und Datei')
plt.ylabel('MAE')
plt.title('Mean Absolute Error für verschiedene Modelle und Dateien')

# Beschrifte die x-Achse mit den Modell- und Dateinamen
model_labels = [
    'Erd - ' + files[0], 'Naive - ' + files[0], 'Hold - ' + files[0],
    'Erd - ' + files[1], 'Naive - ' + files[1], 'Hold - ' + files[1],
    'Erd - ' + files[2], 'Naive - ' + files[2], 'Hold - ' + files[2],
    'Erd - ' + files[3], 'Naive - ' + files[3], 'Hold - ' + files[3]
]

plt.xticks(range(len(model_labels)), model_labels, rotation=45, ha='right')

# Füge die MAE-Werte als Text zu den Balken hinzu
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')

# Erstelle eine Legende
legend_labels = [plt.Rectangle((0,0),1,1, color=colors['Erd']), plt.Rectangle((0,0),1,1, color=colors['Naive']), plt.Rectangle((0,0),1,1, color=colors['Hold'])]
plt.legend(legend_labels, colors.keys())

plt.tight_layout()
plt.show()
