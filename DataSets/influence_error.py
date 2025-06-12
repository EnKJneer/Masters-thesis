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
    t3, t2 = x
    return a * t3 + b * t2 + c

def combined_model_linear_3(x, a, b, c, d):
    x_1, x_2, x_3 = x
    return a * x_1 + b * x_2 + c * x_3 + d

def plot_2d_with_color(x_values, y_values, color_values, label='|v_x + v_y|', title = '2D Plot von pos_x und pos_y mit Farbe', dpi=300, xlabel = 'pos_x', ylabel = 'pos_y'):
    """
    Erstellt einen 2D-Plot mit Linien, deren Farbe basierend auf den color_values bestimmt wird.

    :param x_values: Liste oder Array der x-Werte
    :param y_values: Liste oder Array der y-Werte
    :param color_values: Liste oder Array der Werte, die die Farbe bestimmen
    :param label: Name der Farbskala (Standard: '|v_x + v_y|')
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    """
    # Erstellen des Plots mit höherer Auflösung
    plt.figure(figsize=(10, 6), dpi=dpi)

    # Normalisieren der color_values für den Farbverlauf
    normalized_color_values = (color_values - np.min(color_values)) / (np.max(color_values) - np.min(color_values))

    # Erstellen eines Streudiagramms, um die Farbskala anzuzeigen
    sc = plt.scatter(x_values, y_values, c=color_values, cmap='viridis', s=1)

    # Hinzufügen einer Farbskala
    plt.colorbar(sc, label=label)

    # Beschriftungen und Titel hinzufügen
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{title}')
    #plt.savefig(filename + '.png')
    # Anzeigen des Plots
    plt.show()

# Pfad zu den Dateien
path_data = 'DataFiltered'

# Liste der Dateien
#files = ['AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv',
         #'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']
files = ['AL_2007_T4_Plate_SF_3.csv', 'AL_2007_T4_Plate_Depth_2.csv', 'AL_2007_T4_Plate_Normal_3.csv']
# Anzahl der letzten Datenpunkte, die ausgeschlossen werden sollen
n = 25

# Liste zur Speicherung der MAE-Werte für alle Dateien
mae_values = []

# Iteriere über die Dateien
for file in files:
    # Lade die Daten
    data = pd.read_csv(os.path.join(path_data, file))

    data = data.iloc[n:-n]

    # Berechne die Komponenten der Materialentfernung
    data['mrr_x'] = -data['materialremoved_sim'] * ((data['v_x']) / (np.abs(data['v_x']) + np.abs(data['v_y']) + 1e-10))
    data['mrr_y'] = -data['materialremoved_sim'] * ((data['v_y']) / (np.abs(data['v_x']) + np.abs(data['v_y']) + 1e-10))

    f_x = data['f_x_sim'].values
    f_y = data['f_y_sim'].values
    v_x = data['v_x'].values
    v_y = data['v_y'].values
    a_x = data['a_x'].values
    a_y = data['a_y'].values
    y = data['curr_x'].values
    mrr_x = data['mrr_x'].values
    mrr_y = data['mrr_y'].values

    x_values = data['pos_x'].values
    y_values = data['pos_y'].values

    axis = 'x'
    data[f't2_{axis}'] = data[f'v_{axis}'] ** 2 * np.sign(data[f'v_{axis}'])
    data[f't2_{axis}_s'] = data[f'v_{axis}'] ** 2 * np.sign(data[f'v_{axis}'])
    data[f't3_{axis}'] = data[f'f_{axis}_sim'] * data[f'mrr_{axis}']
    t2 = data[f't2_{axis}'].values
    t3 = data[f't3_{axis}'].values

    # Kombinierte Gleichung mit t3 und sign(v_x)
    initial_params_combined_lin = [1, 1, 1, 1]

    params_combined_lin, _ = curve_fit(combined_model_linear_3, (f_x, np.sign(v_x), a_x), y, p0=initial_params_combined_lin)
    y_pred_combined_lin = combined_model_linear_3((f_x, np.sign(v_x), a_x), *params_combined_lin)
    mae_combined_lin = calculate_mae(y, y_pred_combined_lin)
    mae_values.append(mae_combined_lin)


    # Plotte den zeitlichen Verlauf des Stroms, der Vorhersage und des Fehlers
    loss = y - y_pred_combined_lin
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, y, label='curr_x', color='blue')
    plt.plot(data.index, y_pred_combined_lin, label='Predicted curr_x', color='green')
    plt.plot(data.index, loss, label='error Naive model', color='red')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(f'Zeitlicher Verlauf von Strom, Vorhersage und Fehler für {file}')
    plt.legend()
    plt.show()

    color_values = np.clip(loss, -0.5, 0.5)
    plot_2d_with_color(x_values, y_values, color_values, label='error', title=file + 'sign(v_x)',
                       dpi=300)


    data['loss'] = loss
    # Berechne die Korrelationsmatrix für curr_x
    corr_matrix = data[['loss', 'mrr_x', 'mrr_y', 'materialremoved_sim', 'v_y', 'f_y_sim', 'curr_y', 'v_z', 'f_z_sim', 'curr_z', 'f_sp_sim', 'curr_sp']].corr()

    # Plotte die Korrelationsmatrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'{file}: Korrelationsmatrix für curr_x und andere Komponenten')
    plt.show()

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 1. Plot: loss vs. materialremoved_sim
    ax1 = axs[0]
    ax2 = ax1.twinx()
    ax1.plot(data.index, data['loss'], label='loss', color='red')
    ax2.plot(data.index, data['materialremoved_sim'], label='materialremoved_sim', color='blue')
    ax1.set_ylabel('loss', color='red')
    ax2.set_ylabel('materialremoved_sim', color='blue')
    ax1.set_title(f'{file} - Zeitverlauf: loss & materialremoved_sim')

    # 2. Plot: loss vs. mrr_x
    ax1 = axs[1]
    ax2 = ax1.twinx()
    ax1.plot(data.index, data['loss'], label='loss', color='red')
    ax2.plot(data.index, data['mrr_x'], label='mrr_x', color='green')
    ax1.set_ylabel('loss', color='red')
    ax2.set_ylabel('mrr_x', color='green')
    ax1.set_title(f'{file} - Zeitverlauf: loss & mrr_x')

    # 3. Plot: loss vs. mrr_y
    ax1 = axs[2]
    ax2 = ax1.twinx()
    ax1.plot(data.index, data['loss'], label='loss', color='red')
    ax2.plot(data.index, data['mrr_y'], label='mrr_y', color='purple')
    ax1.set_ylabel('loss', color='red')
    ax2.set_ylabel('mrr_y', color='purple')
    ax1.set_xlabel('Index')
    ax1.set_title(f'{file} - Zeitverlauf: loss & mrr_y')

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    axs[0].scatter(data['materialremoved_sim'], data['loss'], s=5, alpha=0.5, c='blue')
    axs[0].set_xlabel('materialremoved_sim')
    axs[0].set_ylabel('loss')
    axs[0].set_title(f'{file} - Scatter: loss vs. materialremoved_sim')

    axs[1].scatter(data['mrr_x'], data['loss'], s=5, alpha=0.5, c='green')
    axs[1].set_xlabel('mrr_x')
    axs[1].set_ylabel('loss')
    axs[1].set_title(f'{file} - Scatter: loss vs. mrr_x')

    axs[2].scatter(data['mrr_y'], data['loss'], s=5, alpha=0.5, c='purple')
    axs[2].set_xlabel('mrr_y')
    axs[2].set_ylabel('loss')
    axs[2].set_title(f'{file} - Scatter: loss vs. mrr_y')

    plt.tight_layout()
    plt.show()



