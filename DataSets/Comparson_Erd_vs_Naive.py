import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
def combined_model_linear_4(x, a, b, c, d, e):
    x_1, x_2, x_3, x_4 = x
    return a * x_1 + b * x_2 + c * x_3 + d *x_4


# Pfad zu den Dateien
path_data = 'DataFiltered'

# Liste der Dateien
files = ['AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv',
         'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']

# Anzahl der letzten Datenpunkte, die ausgeschlossen werden sollen
n = 25

# Liste zur Speicherung der MAE-Werte für alle Dateien
mae_values = []

# Iteriere über die Dateien
for file in files:
    if '_3' in file:
        # Lade die Daten
        data = pd.read_csv(os.path.join(path_data, file))

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

        # Absolute Werte der Daten
        abs_a_x = np.abs(a_x)

        # Berechnung von Mittelwert und Standardabweichung
        mean = np.mean(abs_a_x)
        std_dev = np.std(abs_a_x)

        # Schwelle für Ausreißer (z.B. 2 Standardabweichungen)
        threshold = 3 * std_dev

        # Identifikation der Ausreißer
        lower_bound = mean - threshold
        upper_bound = mean + threshold

        # Ersetzen der Ausreißer
        a_x_cleaned = np.where(a_x < lower_bound, lower_bound, a_x)
        a_x_cleaned = np.where(a_x_cleaned > upper_bound, upper_bound, a_x_cleaned)
        #a_x = a_x_cleaned
        #plt.figure(figsize=(12, 6))
        #plt.plot(data.index[:-n], a_x, label='a_x', color='blue')
        #plt.xlabel('Index')
        #plt.ylabel('Value')
        #plt.title(f'Zeitlicher Verlauf vder Beschleunigung für {file}')
        #plt.legend()
        #plt.show()

        axis = 'x'
        data[f't1_{axis}'] = data[f'a_{axis}'] * data[f'v_{axis}']
        data[f't2_{axis}'] = data[f'v_{axis}'] ** 2 * np.sign(data[f'v_{axis}'])
        data[f't2_{axis}_s'] = data[f'v_{axis}'] ** 2 * np.sign(data[f'v_{axis}'])
        data[f't3_{axis}'] = data[f'f_{axis}_sim'] * data['materialremoved_sim'] # data[f'mrr_{axis}']
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

        """ # Kombinierte Gleichung mit t3 und sign(v_x)
        initial_params_combined_lin = [1, 1, 1, 1, 1]


        def sign_hold(v_x):
            signs = np.sign(v_x)
            # Finde Indizes wo Vorzeichen != 0
            nonzero_idx = np.nonzero(signs)[0]

            if len(nonzero_idx) == 0:
                return signs  # Alle Werte sind 0

            # Für jeden Index finde den letzten gültigen Vorzeichen-Index
            indices = np.searchsorted(nonzero_idx, np.arange(len(signs)), side='right') - 1
            indices = np.clip(indices, 0, len(nonzero_idx) - 1)

            result = signs.copy()
            # Nur Nullstellen ersetzen
            zero_mask = (signs == 0)
            valid_replacement = indices >= 0

            result[zero_mask & valid_replacement] = signs[nonzero_idx[indices[zero_mask & valid_replacement]]]

            return result

        params_combined_lin, _ = curve_fit(combined_model_linear_4, (f_x, np.sign(v_x), sign_hold(v_x), a_x), y, p0=initial_params_combined_lin)
        y_pred_combined_lin = combined_model_linear_4((f_x, np.sign(v_x), sign_hold(v_x), a_x), *params_combined_lin)
        mae_combined_lin = calculate_mae(y, y_pred_combined_lin)
        mae_values.append(mae_combined_lin)

        # Plotte den zeitlichen Verlauf des Stroms, der Vorhersage und des Fehlers
        error = y - y_pred_combined_lin
        plt.figure(figsize=(12, 6))
        plt.plot(data.index[:-n], y, label='curr_x', color='blue')
        plt.plot(data.index[:-n], y_pred_combined_lin, label='Predicted curr_x', color='green')
        plt.plot(data.index[:-n], error, label='error t3 & sign_hold(v_x)', color='red')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Zeitlicher Verlauf von Strom, Vorhersage und Fehler für {file}')
        plt.legend()
        plt.show()"""

# Erstelle einen Barplot für die MAE-Werte
plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(mae_values)), mae_values, color=['blue', 'green', 'blue', 'green', 'blue', 'green', 'blue', 'green'])
plt.xlabel('Modell und Datei')
plt.ylabel('MAE')
plt.title('Mean Absolute Error für verschiedene Modelle und Dateien')

# Beschrifte die x-Achse mit den Modell- und Dateinamen
plt.xticks(range(len(mae_values)), ['t3 & t2 - ' + files[0], 't3 & sign(v_x) - ' + files[0],
           't3 & t2 - ' + files[1], 't3 & sign(v_x) - ' + files[1],
           't3 & t2 - ' + files[2], 't3 & sign(v_x) - ' + files[2],
           't3 & t2 - ' + files[3], 't3 & sign(v_x) - ' + files[3]], rotation=45)

# Füge die MAE-Werte als Text zu den Balken hinzu
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')

plt.show()
