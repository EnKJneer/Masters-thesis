import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from scipy.optimize import minimize_scalar
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def mse_random_forest_offset(df_target, df_source, target_cols, source_cols, max_shift):
    if len(df_target) < 100 and len(df_source) < 100:
        print("Warnung: Zu wenige saubere Datenpunkte für multivariate Regression")
        return 0

    X_target = df_target[target_cols].values
    y_source = df_source[source_cols].values

    for i in range(X_target.shape[1]):
        X_target[:, i] = apply_lowpass_filter(X_target[:, i], 0.1, 4)
    for i in range(y_source.shape[1]):
        y_source[:, i] = apply_lowpass_filter(y_source[:, i], 0.1, 4)

    def calculate_regression_mse(offset, X_target, y_source, step = 1):
        offset = int(offset)
        if offset == 0:
            X_aligned = X_target
            y_aligned = y_source
        elif offset > 0:
            if offset >= len(X_target):
                return np.inf
            X_aligned = X_target[offset:]
            y_aligned = y_source[:len(X_aligned)]
        else:  # offset < 0
            if -offset >= len(y_source):
                return np.inf
            X_aligned = X_target[:len(X_target) + offset]
            y_aligned = y_source[-offset:]

        min_len = min(len(X_aligned), len(y_aligned))
        if min_len < 10:
            return np.inf

        X_aligned = X_aligned[:min_len:step]
        y_aligned = y_aligned[:min_len:step]

        try:
            reg = RandomForestRegressor(n_estimators=10, n_jobs=-1)
            reg.fit(X_aligned, y_aligned)
            y_pred = reg.predict(X_aligned)
            mse = mean_squared_error(y_aligned, y_pred)
            r2 = reg.score(X_aligned, y_aligned)
            if r2 < 0:
                r2 = 0
            combined_score = mse * (1 - r2)
            return combined_score
        except:
            return np.inf
    print('Starte grobe Ausrichtung')
    offsets = range(-max_shift, max_shift + 1, 20)
    scores = [calculate_regression_mse(offset, X_target, y_source, 100) for offset in offsets]
    best_offset = offsets[scores.index(min(scores))]
    print(f'Offset: {best_offset}')
    print('Starte fein Ausrichtung')
    offsets_highres = range(best_offset - 11, best_offset + 11, 1)
    scores_highres = [calculate_regression_mse(offset, X_target, y_source) for offset in offsets_highres]
    best_offset = offsets_highres[scores_highres.index(min(scores_highres))]
    print(f'Offset: {best_offset}')
    return best_offset

def shift_time_series(df_source, df_target, offset, f_cols):
    df_source_shifted = df_source.copy()
    df_target_shifted = df_target.copy()
    df_source_shifted = df_source_shifted[f_cols]

    if offset > 0:
        if offset >= len(df_target_shifted):
            return np.inf
        df_target_shifted = df_target_shifted[offset:].reset_index(drop=True)
        df_source_shifted = df_source_shifted[:len(df_target_shifted)].reset_index(drop=True)
    else:  # offset < 0
        if -offset >= len(df_source_shifted):
            return np.inf
        df_target_shifted = df_target_shifted[:len(df_target_shifted) + offset].reset_index(drop=True)
        df_source_shifted = df_source_shifted[-offset:].reset_index(drop=True)

    # Kombinieren der DataFrames
    combined_df = pd.concat([df_target_shifted, df_source_shifted], axis=1).reset_index(drop=True)
    return combined_df

def butter_lowpass(cutoff, order, nyq_freq=0.5):
    normal_cutoff = cutoff / nyq_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def apply_lowpass_filter(data, cutoff, order):
    b, a = butter_lowpass(cutoff, order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def plot_time_series(data, title, dpi=300, labels=None, ylabel='curr_x'):
    """
    Erstellt einen Zeitverlaufsplot.
    :param data: DataFrame mit den Daten
    :param title: Titel des Plots
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    :param labels: Liste der Labels für die zu plottenden Daten
    :param ylabel: Label für die zweite y-Achse
    """
    if not isinstance(labels, list):
        labels = [labels]
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=dpi)
    for label in labels:
        ax1.plot(data.index, data[label], label=label)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Werte')
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    # Zweite y-Achse für ylabel
    if ylabel:
        ax2 = ax1.twinx()
        ax2.plot(data.index, data[ylabel], label=ylabel, color='tab:red')
        ax2.set_ylabel(ylabel)
        ax2.legend(loc='upper right')
    plt.show()

# Laden der Daten
path_data_sin = 'RawDataSinumerik'
path_data_dt = 'RawDataDT9836'
output_path = 'MergedData'
files = os.listdir(path_data_sin)

f_s = 500
for file in files:
    if file.endswith('.csv'):
        name = file.replace('.csv', '')
        print(f'### {name} Anfang ###')

        df_sin = pd.read_csv(os.path.join(path_data_sin, file))
        df_dt = pd.read_csv(os.path.join(path_data_dt, file))

        # Bestimmung des zeitlichen Versatzes
        max_shift_seconds = 5
        max_shift = int(max_shift_seconds * f_s)

        ref_signals = ['curr_x', 'curr_y', 'curr_z', 'curr_sp', 'a_x', 'a_y', 'a_z', 'a_sp', 'v_x', 'v_y', 'v_z', 'v_sp']
        available_refs = [col for col in ref_signals if col in df_sin.columns]
        source_names = ['f_x', 'f_y', 'f_z']
        #for name in source_names:
        #    df_dt[name] = apply_lowpass_filter(df_dt[name], 249, 4)
        offset_mv = mse_random_forest_offset(df_sin, df_dt, target_cols=available_refs, source_cols=source_names, max_shift=max_shift)
        #print(f'Offset: {offset_mv}')

        # Verschieben der Daten
        # Verschieben der Daten
        data = shift_time_series(df_dt, df_sin, offset_mv, ['f_x', 'f_y', 'f_z'])
        data = data.dropna().reset_index(drop=True)

        # Speichern der ausgerichteten Daten
        output_file = os.path.join(output_path, file)
        data.to_csv(output_file, index=False)
        print(f"Gespeichert: {output_file}")
        plot_time_series(data, name, labels='curr_x', dpi=300, ylabel='f_x')
        print(f'### {file} Ende ###')
