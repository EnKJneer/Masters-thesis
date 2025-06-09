import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns


# === Modellfunktionen ===
def combined_model_linear_3(x, a, b, c, d):
    x_1, x_2, x_3 = x
    return a * x_1 + b * x_2 + c * x_3 + d

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def sigmoid_basic(x):
    return 1 / (1 + np.exp(-x))


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

def combined_model_sigmoid_theta(x, a, b, c, d, theta):
    x_1, x_2, x_3 = x
    sigmoid_v = 1 / (1 + np.exp(-(x_2 - theta)))
    return a * x_1 + b * sigmoid_v + c * x_3 + d
# === Plots ===
def plot_error_over_time(index, y_true, y_pred, label, title):
    error = y_true - y_pred
    plt.figure(figsize=(12, 6))
    plt.plot(index, y_true, label='curr_x', color='blue')
    plt.plot(index, y_pred, label='Predicted', color='green')
    plt.plot(index, error, label='Error', color='red')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title + f' ({label})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return error

def plot_2d_with_color(x, y, color, label='Error', title='2D Plot', dpi=300, xlabel='pos_x', ylabel='pos_y'):
    plt.figure(figsize=(10, 6), dpi=dpi)
    norm_color = (color - np.min(color)) / (np.max(color) - np.min(color) + 1e-10)
    sc = plt.scatter(x, y, c=color, cmap='viridis', s=1)
    plt.colorbar(sc, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# === MAE und Modellanpassung ===
def fit_and_evaluate_model(x_tuple, y_true, model_func, label, index, pos_x, pos_y, clip_val=0.5, p0=None):
    if p0 is None:
        p0 = [1] * 4  # Standard bei 4 Parametern
    params, _ = curve_fit(model_func, x_tuple, y_true, p0=p0)
    y_pred = model_func(x_tuple, *params)
    mae = calculate_mae(y_true, y_pred)

    error = plot_error_over_time(index, y_true, y_pred, label=label, title='Stromvorhersage & Fehler')
    plot_2d_with_color(pos_x, pos_y, np.clip(error, -clip_val, clip_val), label='Fehler', title=label)

    return mae

# === Hauptprozess ===
def process_all_files(path_data, files, n=25):
    mae_results = []
    mae_labels = []

    for file in files:
        print(f'Verarbeite Datei: {file}')
        data = pd.read_csv(os.path.join(path_data, file))
        axis = 'x'

        # Vorverarbeitung
        data['mrr_x'] = data['materialremoved_sim'] * (
                    np.abs(data['v_x']) / (np.abs(data['v_x']) + np.abs(data['v_y']) + 1e-10))
        data['t2_x'] = data['v_x'] ** 2 * np.sign(data['v_x'])
        data['t3_x'] = data['f_x_sim'] * data['mrr_x']

        f_x = data['f_x_sim'].iloc[:-n].values
        a_x = data['a_x'].iloc[:-n].values
        v_x = data['v_x'].iloc[:-n].values
        y = data['curr_x'].iloc[:-n].values
        idx = data.index[:-n]
        pos_x = data['pos_x'].iloc[:-n].values
        pos_y = data['pos_y'].iloc[:-n].values

        epsilon = 1e-8
        v_x_normalized = v_x / (np.abs(v_x) + epsilon)
        model_inputs = {
            'sign(v_x)': ((f_x, np.sign(v_x), a_x), combined_model_linear_3),
            'sign_hold(v_x)': ((f_x, sign_hold(v_x), a_x), combined_model_linear_3),
            #'sigmoid(v_x)': ((f_x, sigmoid_basic(v_x), a_x), combined_model_linear_3),
            'sigmoid(v_x - theta)': ((f_x, v_x, a_x), combined_model_sigmoid_theta)
        }

        for label, (inputs, model_func) in model_inputs.items():
            p0 = [1] * (model_func.__code__.co_argcount - 1)  # Initialwerte automatisch
            mae = fit_and_evaluate_model(inputs, y, model_func, f'{label} - {file}', idx, pos_x, pos_y, p0=p0)
            mae_results.append(mae)
            mae_labels.append(f'{label} - {file}')

    return mae_results, mae_labels

# === Ergebnisse visualisieren ===
def plot_mae_results(mae_values, labels):
    import matplotlib.colors as mcolors

    # Modelltypen aus Labels extrahieren (z. B. 'sign(v_x)' aus 'sign(v_x) - AL_...')
    model_types = [label.split(' - ')[0] for label in labels]
    unique_models = sorted(set(model_types))

    # Farben zuweisen
    color_palette = sns.color_palette("Set2", len(unique_models))
    model_to_color = {model: color for model, color in zip(unique_models, color_palette)}

    # Farben entsprechend den Modelltypen in Reihenfolge der Labels
    bar_colors = [model_to_color[model] for model in model_types]

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(mae_values)), mae_values, color=bar_colors)
    plt.xticks(range(len(mae_values)), labels, rotation=45, ha='right')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('Modellvergleich: MAE pro Datei')

    # MAE-Werte beschriften
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom')

    # Legende für die Modelltypen
    legend_patches = [plt.Line2D([0], [0], marker='s', color='w', label=model,
                                  markerfacecolor=model_to_color[model], markersize=10)
                      for model in unique_models]
    plt.legend(handles=legend_patches, title='Modelltyp', loc='upper right')

    plt.tight_layout()
    plt.show()

# === Main ===
if __name__ == '__main__':
    path_data = 'DataFiltered'
    files = ['AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv',
             'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']

    mae_values, mae_labels = process_all_files(path_data, files)
    plot_mae_results(mae_values, mae_labels)

    # === Mittelwert der MAEs pro Modelltyp berechnen ===
    from collections import defaultdict

    mae_per_model = defaultdict(list)

    for label, mae in zip(mae_labels, mae_values):
        model_type = label.split(' - ')[0]
        mae_per_model[model_type].append(mae)

    avg_mae_per_model = {model: np.mean(maes) for model, maes in mae_per_model.items()}
    best_model = min(avg_mae_per_model, key=avg_mae_per_model.get)

    print('\n=== Durchschnittlicher MAE pro Modelltyp ===')
    for model, avg_mae in avg_mae_per_model.items():
        print(f'{model}: {avg_mae:.4f}')

    print(f'\n=== Bestes Modell im Mittel über alle Dateien: {best_model} (MAE = {avg_mae_per_model[best_model]:.4f}) ===')
