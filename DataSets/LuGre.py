import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# --- LuGre Modell Klasse (wie vorher) ---
class LuGreFriction:
    def __init__(self, sigma_0, sigma_1, sigma_2, F_s, F_c, v_s):
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.F_s = F_s
        self.F_c = F_c
        self.v_s = v_s
        self.z = 0.0

    def g(self, v):
        return self.F_c + (self.F_s - self.F_c) * np.exp(-(v / self.v_s)**2)

    def step(self, v, dt):
        dz = v - (self.sigma_0 * np.abs(v) / self.g(v)) * self.z
        self.z += dz * dt
        F_friction = self.sigma_0 * self.z + self.sigma_1 * dz + self.sigma_2 * v
        return F_friction

# --- Kombiniertes Modell für curve_fit ---
def model_lugre(x_tuple, a, b, c, d, sigma_0, sigma_1, sigma_2, F_s, F_c, v_s):
    f_x, sign_vx, a_x, v_x = x_tuple
    dt = 0.02  # Samplingzeit 50 Hz
    lugre_model = LuGreFriction(sigma_0, sigma_1, sigma_2, F_s, F_c, v_s)
    lugre_forces = []

    # LuGre über Zeitschritte iterieren
    for v in v_x:
        F_lugre = lugre_model.step(v, dt)
        lugre_forces.append(F_lugre)

    lugre_forces = np.array(lugre_forces)

    # Lineares Modell (4 Parameter)
    linear_part = a * f_x + b * sign_vx + c * a_x + d

    return linear_part + lugre_forces
def model_lugre_wrapper(x, a, b, c, d, sigma_0, sigma_1, sigma_2, F_s, F_c, v_s):
    # x ist ein 2D-Array mit z.B. Form (4, n_samples)
    f_x = x[0]
    sign_vx = x[1]
    a_x = x[2]
    v_x = x[3]
    return model_lugre((f_x, sign_vx, a_x, v_x), a, b, c, d, sigma_0, sigma_1, sigma_2, F_s, F_c, v_s)
# --- MAE Berechnung ---
def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# --- Plot Funktion für Zeitverlauf & Fehler ---
def plot_error_over_time(index, y_true, y_pred, label, title):
    error = y_true - y_pred
    plt.figure(figsize=(12, 6))
    plt.plot(index, y_true, label='curr_x (wahr)', color='blue')
    plt.plot(index, y_pred, label='Vorhersage', color='green')
    plt.plot(index, error, label='Fehler', color='red')
    plt.xlabel('Index')
    plt.ylabel('Strom')
    plt.title(title + f' ({label})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return error

# --- Fehler im Raum plotten ---
def plot_2d_with_color(x, y, color, label='Fehler', title='2D Plot', dpi=300, xlabel='pos_x', ylabel='pos_y'):
    plt.figure(figsize=(10, 6), dpi=dpi)
    norm_color = (color - np.min(color)) / (np.max(color) - np.min(color) + 1e-10)
    sc = plt.scatter(x, y, c=color, cmap='viridis', s=1)
    plt.colorbar(sc, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# --- Fit & Evaluate Funktion ---
def fit_and_evaluate_combined_lugre(x_tuple, y_true, index, pos_x, pos_y, label='Combined linear + LuGre'):
    # Anfangswerte: 4 für linear + 6 für LuGre
    p0 = [1, 1, 1, 0] + [1e5, 100, 0.5, 10, 5, 0.01]
    bounds_lower = [-np.inf, -np.inf, -np.inf, -np.inf, 1e3, 1, 0, 0, 0, 1e-4]
    bounds_upper = [np.inf, np.inf, np.inf, np.inf, 1e7, 1e3, 10, 100, 50, 0.1]

    x_data = np.vstack(x_tuple)  # Form (4, n_samples)
    params_opt, _ = curve_fit(model_lugre_wrapper, x_data, y_true, p0=p0, bounds=(bounds_lower, bounds_upper),
                              maxfev=10000)

    y_pred = model_lugre_wrapper(x_tuple, *params_opt)
    mae = calculate_mae(y_true, y_pred)

    # Plots
    error = plot_error_over_time(index, y_true, y_pred, label, 'Stromvorhersage & Fehler')
    plot_2d_with_color(pos_x, pos_y, error, label='Fehler', title=label)

    # Parameter ausgeben
    print(f'\n{label} - Optimierte Parameter:')
    print(f'Linear: a={params_opt[0]:.3f}, b={params_opt[1]:.3f}, c={params_opt[2]:.3f}, d={params_opt[3]:.3f}')
    print(f'LuGre: sigma_0={params_opt[4]:.2e}, sigma_1={params_opt[5]:.2f}, sigma_2={params_opt[6]:.2f}, '
          f'F_s={params_opt[7]:.2f}, F_c={params_opt[8]:.2f}, v_s={params_opt[9]:.4f}')
    print(f'MAE: {mae:.4f}')

    return mae

# === Beispiel: Nutzung im Hauptprozess ===
def process_files_with_combined_model(path_data, files):
    mae_results = []
    mae_labels = []

    for file in files:
        print(f'Verarbeite Datei: {file}')
        data = pd.read_csv(os.path.join(path_data, file))
        n = 25  # wie in deinem Code

        f_x = data['f_x_sim'].iloc[:-n].values
        v_x = data['v_x'].iloc[:-n].values
        a_x = data['a_x'].iloc[:-n].values
        y_true = data['curr_x'].iloc[:-n].values
        index = data.index[:-n]
        pos_x = data['pos_x'].iloc[:-n].values
        pos_y = data['pos_y'].iloc[:-n].values

        x_tuple = (f_x, np.sign(v_x), a_x, v_x)

        mae = fit_and_evaluate_combined_lugre(x_tuple, y_true, index, pos_x, pos_y, label=f'Combined linear + LuGre - {file}')
        mae_results.append(mae)
        mae_labels.append(f'Combined linear + LuGre - {file}')

    return mae_results, mae_labels

def combined_model_linear_3(x, a, b, c, d):
    x_1, x_2, x_3 = x
    return a * x_1 + b * x_2 + c * x_3 + d
# Pfad zu den Dateien
path_data = 'DataFiltered'

# Liste der Dateien
files = ['AL_2007_T4_Plate_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']
#, 'AL_2007_T4_Gear_Normal_3.csv',         'S235JR_Gear_Normal_3.csv',

# Liste zur Speicherung der MAE-Werte für alle Dateien
mae_values = []

# Iteriere über die Dateien
for file in files:
    if '_3' in file:

        # Lade die Daten
        data = pd.read_csv(os.path.join(path_data, file))
        n = 25  # Anzahl der letzten Datenpunkte, die ausgeschlossen werden sollen

        # Berechne mrr_x und mrr_y
        data['mrr_x'] = data['materialremoved_sim'] * (np.abs(data['v_x']) / (np.abs(data['v_x']) + np.abs(data['v_y']) + 1e-10))
        data['mrr_y'] = data['materialremoved_sim'] * (np.abs(data['v_y']) / (np.abs(data['v_x']) + np.abs(data['v_y']) + 1e-10))

        f_x = data['f_x_sim'].iloc[:-n].values
        v_x = data['v_x'].iloc[:-n].values
        a_x = data['a_x'].iloc[:-n].values
        y = data['curr_x'].iloc[:-n].values

        # Naive Modell (wie vorher)
        initial_params_naive = [1, 1, 1, 1]
        params_naive, _ = curve_fit(combined_model_linear_3, (a_x, f_x, np.sign(v_x)), y, p0=initial_params_naive)
        y_pred_naive = combined_model_linear_3((a_x, f_x, np.sign(v_x)), *params_naive)
        mae_naive = calculate_mae(y, y_pred_naive)
        mae_values.append(mae_naive)
        print(f'Model Naive: {params_naive}')

        plt.figure(figsize=(12, 6))
        plt.plot(data.index[:-n], y, label='curr_x', color='blue')
        plt.plot(data.index[:-n], y_pred_naive, label='Predicted curr_x (Naive)', color='green')
        plt.plot(data.index[:-n], y - y_pred_naive, label='loss Naive', color='red')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Naive Modell - Strom, Vorhersage und Fehler für {file}')
        plt.legend()
        plt.show()

        # LuGre Modell (Beispiel, Parameter und Modell müssen definiert sein)

        # Hier nur als Template, bitte an dein LuGre-Modell anpassen
        initial_params_lugre = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Beispielwerte, anpassen!
        x_data = (f_x, np.sign(v_x), a_x, v_x)
        params_lugre, _ = curve_fit(model_lugre_wrapper, x_data, y, p0=initial_params_lugre)
        y_pred_lugre = model_lugre_wrapper(x_data, *params_lugre)
        mae_lugre = calculate_mae(y, y_pred_lugre)
        mae_values.append(mae_lugre)
        print(f'Model LuGre: {params_lugre}')

        plt.figure(figsize=(12, 6))
        plt.plot(data.index[:-n], y, label='curr_x', color='blue')
        plt.plot(data.index[:-n], y_pred_lugre, label='Predicted curr_x (LuGre)', color='purple')
        plt.plot(data.index[:-n], y - y_pred_lugre, label='loss LuGre', color='orange')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'LuGre Modell - Strom, Vorhersage und Fehler für {file}')
        plt.legend()
        plt.show()

# Plot für MAE Werte

plt.figure(figsize=(15, 8))

colors = {
    'Naive': 'green',
    'LuGre': 'purple'
}

# Balkenfarben: je zwei Balken pro Datei (Naive, LuGre)
bar_colors = [colors['Naive'], colors['LuGre']] * len(files)

bars = plt.bar(range(len(mae_values)), mae_values, color=bar_colors)
plt.xlabel('Modell und Datei')
plt.ylabel('MAE')
plt.title('Mean Absolute Error für Naive und LuGre Modelle und Dateien')

# Beschriftungen
model_labels = []
for file in files:
    model_labels.append('Naive - ' + file)
    model_labels.append('LuGre - ' + file)

plt.xticks(range(len(model_labels)), model_labels, rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')

legend_labels = [plt.Rectangle((0, 0), 1, 1, color=colors[key]) for key in colors]
plt.legend(legend_labels, colors.keys())

plt.tight_layout()
plt.show()