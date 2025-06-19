import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# --- LuGre Modell Klasse ---
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
        eps = 1e-12
        return self.F_c + (self.F_s - self.F_c) * np.exp(-(v / self.v_s)**2) + eps

    def dz_fn(self, z, v):
        return v - (self.sigma_0 * np.abs(v) / self.g(v)) * z

    def step(self, v, dt):
        # Runge-Kutta 4. Ordnung
        k1 = self.dz_fn(self.z, v)
        k2 = self.dz_fn(self.z + 0.5 * dt * k1, v)
        k3 = self.dz_fn(self.z + 0.5 * dt * k2, v)
        k4 = self.dz_fn(self.z + dt * k3, v)

        self.z += (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.z = np.clip(self.z, -1e6, 1e6)

        v = np.clip(v, -1e3, 1e3)
        F_friction = self.sigma_0 * self.z + self.sigma_1 * self.dz_fn(self.z, v) + self.sigma_2 * v
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
    linear_part = a * f_x + c * a_x + d
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

def combined_model_linear_3(x, a, b, c, d):
    x_1, x_2, x_3 = x
    return a * x_1 + b * x_2 + c * x_3 + d
# Pfad zu den Dateien
path_data = 'DataFiltered'

# Liste der Dateien
files = ['AL_2007_T4_Plate_Normal_3.csv']#, 'AL_2007_T4_Gear_Normal_3.csv', 'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']
files_test = ['AL_2007_T4_Plate_Depth_3.csv', 'AL_2007_T4_Gear_Normal_3.csv', 'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']

# Liste zur Speicherung der MAE-Werte für alle Dateien
mae_values = []

# Iteriere über die Dateien
for file in files:

    # Lade die Daten
    data = pd.read_csv(os.path.join(path_data, file))
    n = 25  # Anzahl der letzten Datenpunkte, die ausgeschlossen werden sollen

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
    params_lugre, _ = curve_fit(model_lugre_wrapper, x_data, y, p0=initial_params_lugre, maxfev=10000)
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
# === Test auf neuen Dateien mit den trainierten LuGre-Parametern ===
mae_values_test = []

# Nutze die zuletzt gelernten LuGre-Parameter aus Training
lugre_params_trained = params_lugre  # aus letztem Trainingslauf

for file in files_test:
    print(f'\nTeste auf Datei: {file}')
    data = pd.read_csv(os.path.join(path_data, file))
    n = 25

    f_x = data['f_x_sim'].iloc[:-n].values
    v_x = data['v_x'].iloc[:-n].values
    a_x = data['a_x'].iloc[:-n].values
    y = data['curr_x'].iloc[:-n].values

    x_data = (f_x, np.sign(v_x), a_x, v_x)

    # Naive Modell wie vorher
    y_pred_naive = combined_model_linear_3((a_x, f_x, np.sign(v_x)), *params_naive)
    mae_naive = calculate_mae(y, y_pred_naive)
    mae_values_test.append(mae_naive)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index[:-n], y, label='curr_x', color='blue')
    plt.plot(data.index[:-n], y_pred_naive, label='Naive Vorhersage', color='green')
    plt.plot(data.index[:-n], y - y_pred_naive, label='Fehler Naive', color='red')
    plt.title(f'Naive Modell - Testdatei: {file}')
    plt.legend()
    plt.grid()
    plt.show()

    # LuGre Vorhersage mit gelernten Parametern
    y_pred_lugre = model_lugre_wrapper(x_data, *lugre_params_trained)
    mae_lugre = calculate_mae(y, y_pred_lugre)
    mae_values_test.append(mae_lugre)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index[:-n], y, label='curr_x', color='blue')
    plt.plot(data.index[:-n], y_pred_lugre, label='LuGre Vorhersage', color='purple')
    plt.plot(data.index[:-n], y - y_pred_lugre, label='Fehler LuGre', color='orange')
    plt.title(f'LuGre Modell - Testdatei: {file}')
    plt.legend()
    plt.grid()
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

# --- MAE Balkendiagramm (Train + Test) ---
plt.figure(figsize=(15, 8))

all_mae = mae_values + mae_values_test
all_labels = []

# Labels für Training
for file in files:
    all_labels.append('Naive - ' + file)
    all_labels.append('LuGre - ' + file)

# Labels für Test
for file in files_test:
    all_labels.append('Naive Test - ' + file)
    all_labels.append('LuGre Test - ' + file)

bar_colors = [colors['Naive'], colors['LuGre']] * (len(files) + len(files_test))

bars = plt.bar(range(len(all_mae)), all_mae, color=bar_colors)
plt.xlabel('Modell und Datei')
plt.ylabel('MAE')
plt.title('MAE Vergleich für Trainings- und Testdaten')
plt.xticks(range(len(all_labels)), all_labels, rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')

plt.legend(legend_labels, colors.keys())
plt.tight_layout()
plt.show()