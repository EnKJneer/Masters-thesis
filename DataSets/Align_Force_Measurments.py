import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import minimize_scalar
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

from Models.model_random_forest import RandomForestModel

warnings.filterwarnings('ignore')

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

def mse_random_forest_offset(data, target_cols=['curr_x', 'curr_y', 'curr_z', 'curr_sp', 'a_x', 'a_y', 'a_z', 'a_sp', 'v_x', 'v_y', 'v_z', 'v_sp'], source_cols=['f_x_butterworth', 'f_y_butterworth', 'f_z_butterworth'], max_shift=500):
    """
    MSE-Optimierung mit Random Forest.
    Verwendet mehrere Signale zur Bestimmung des optimalen Offsets.

    Parameters:
    data: DataFrame mit den Signalen
    target_cols: Liste der Referenzsignale
    source_cols: Zu verschiebendes Signale
    max_shift: Maximaler Verschiebungsbereich
    """
    # Prüfe welche Spalten verfügbar sind
    available_cols = [col for col in target_cols if col in data.columns]

    # Erstelle saubere Daten ohne NaN
    clean_data = data[available_cols + source_cols].dropna()

    if len(clean_data) < 100:
        print("Warnung: Zu wenige saubere Datenpunkte für multivariate Regression")
        return 0, [], []

    X_target = clean_data[available_cols].values
    y_source = clean_data[source_cols].values

    max_shift = min(max_shift, len(clean_data) // 3)

    def calculate_regression_mse(offset):
        offset = int(offset)

        if offset == 0:
            X_aligned = X_target
            y_aligned = y_source
        elif offset > 0:
            # source startet später
            if offset >= len(X_target):
                return np.inf
            X_aligned = X_target[offset:]
            y_aligned = y_source[:len(X_aligned)]
        else:  # offset < 0
            # source startet früher
            if -offset >= len(y_source):
                return np.inf
            X_aligned = X_target[:len(X_target) + offset]
            y_aligned = y_source[-offset:]

        min_len = min(len(X_aligned), len(y_aligned))
        if min_len < 10:  # Mindestens 10 Punkte für Regression
            return np.inf

        X_aligned = X_aligned[:min_len]
        y_aligned = y_aligned[:min_len]

        try:
            # Lineare Regression
            reg = RandomForestRegressor(n_estimators=10, n_jobs=-1)
            reg.fit(X_aligned, y_aligned)
            y_pred = reg.predict(X_aligned)

            # MSE zwischen Vorhersage und tatsächlichen Werten
            mse = mean_squared_error(y_aligned, y_pred)
            print(f"MSE: {mse}")
            # Zusätzlich: R²-Score als Gewichtung (höherer R² = bessere Anpassung)
            r2 = reg.score(X_aligned, y_aligned)

            # Kombiniere MSE und R² (niedrigere MSE und höherer R² sind besser)
            # Invertiere R² für Minimierung: (1 - r2) * mse
            combined_score = mse * (2 - r2)  # r2 zwischen 0 und 1, also (2-r2) zwischen 1 und 2

            return combined_score
        except:
            return np.inf

    # Optimiere den Offset
    # Berechne Scores für Visualisierung
    offsets = range(-max_shift, max_shift + 1, 20)
    scores = [calculate_regression_mse(offset) for offset in offsets]
    best_offset = offsets[scores.index(min(scores))]

    offsets_highres = range(best_offset - 11, best_offset + 11, 1)
    scores_highres = [calculate_regression_mse(offset) for offset in offsets_highres]
    best_offset = offsets_highres[scores_highres.index(min(scores_highres))]

    return best_offset, offsets, scores

def apply_offset_to_signal(signal, offset):
    """
    Wendet einen Offset auf ein Signal an.
    Positiver Offset: Signal startet später (füge NaN am Anfang hinzu)
    Negativer Offset: Signal startet früher (schneide Anfang ab)
    """
    if offset > 0:
        # Signal startet später - füge NaN am Anfang hinzu
        return np.concatenate([np.full(offset, np.nan), signal])
    elif offset < 0:
        # Signal startet früher - schneide Anfang ab
        return signal[-offset:]
    else:
        return signal.copy()

def shift_time_series(data, n, f_cols, curr_col):
    data_2 = data.copy()

    if n > 0:
        # Verschiebe f_x um n nach rechts
        for col in f_cols:
            shifted = data[col].iloc[n:].reset_index(drop=True)
            shifted = pd.concat([shifted, pd.Series([np.nan] * n)], ignore_index=True)
            data_2[col] = shifted
    elif n < 0:
        # Verschiebe curr_x um |n| nach rechts
        shifted_curr = data[curr_col].iloc[abs(n):].reset_index(drop=True)
        shifted_curr = pd.concat([shifted_curr, pd.Series([np.nan] * abs(n))], ignore_index=True)
        data_2[curr_col] = shifted_curr

    return data_2

def finde_nahe_nullbereiche(serie, min_laenge=100, epsilon=0.5):
    """
    Findet zusammenhängende Bereiche, in denen der Absolutwert der Serie < epsilon ist.

    Args:
        serie (pd.Series): Serie mit numerischen Werten.
        min_laenge (int): Minimale Länge des Bereichs, um berücksichtigt zu werden.
        epsilon (float): Toleranzschwelle für "nahe Null".

    Returns:
        Liste von dicts mit 'start', 'ende', 'laenge', 'mitte'
    """
    ist_nahe_null = serie.abs() < epsilon
    gruppen = (ist_nahe_null != ist_nahe_null.shift()).cumsum()

    ergebnisse = []
    for gruppe, gruppe_daten in serie.groupby(gruppen):
        if ist_nahe_null.loc[gruppe_daten.index[0]]:
            start = gruppe_daten.index[0]
            ende = gruppe_daten.index[-1]
            laenge = ende - start + 1
            if laenge >= min_laenge:
                mitte = start + laenge // 2
                ergebnisse.append({
                    'start': start,
                    'ende': ende,
                    'laenge': laenge,
                    'mitte': mitte
                })
    return ergebnisse

def butter_lowpass(cutoff, order, nyq_freq=0.5):
    normal_cutoff = cutoff / nyq_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def apply_lowpass_filter(data, cutoff, order):
    b, a = butter_lowpass(cutoff, order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

output_path = 'OldData_Aligned'
path_data = 'Old_CombinedData'
files = os.listdir(path_data)

for file in files:
    if '.csv' in file:
        name = file.replace('.csv', '')

        print(f'### {name} Anfang###')

        # Butterworth-Filter-Parameter
        cutoff = 0.1  # Normalisierte Grenzfrequenz (z.B. 0.1 bedeutet 10% der Nyquist-Frequenz)
        order = 4  # Filterordnung

        data = pd.read_csv(f'{path_data}/{file}')
        axes = ['x', 'y', 'z']
        for axis in axes:
            data[f'f_{axis}'] = data[f'f_{axis}'].apply(lambda x: np.NAN if np.abs(x) > 1000 else x)
            data = data.dropna().reset_index(drop=True)
            # Berechnung der Standardabweichung
            std_dev = data[f'f_{axis}'].std()
            # Berechnung des Mittelwerts
            mean = data[f'f_{axis}'].mean()
            # Festlegung der Grenzen basierend auf der Standardabweichung
            lower_bound = mean - 4 * std_dev
            upper_bound = mean + 4 * std_dev
            # Begrenzen der Daten
            data[f'f_{axis}'] = np.clip(data[f'f_{axis}'], lower_bound, upper_bound)
            # Butterworth-Filter auf die Daten anwenden
            data[f'f_{axis}_butterworth'] = apply_lowpass_filter(data[f'f_{axis}'], cutoff, order)

        # Bestimme zeitlichen versatz.
        max_shift_seconds = 5
        sampling_rate = 50
        max_shift = int(max_shift_seconds * sampling_rate)

        # Bestimme verfügbare Referenzsignale
        ref_signals = ['curr_x', 'curr_y', 'a_x', 'a_y', 'a_z', 'a_sp', 'v_x', 'v_y', 'v_z', 'v_sp']
        available_refs = [col for col in ref_signals if col in data.columns]

        source_names = ['f_x_butterworth', 'f_y_butterworth', 'f_z_butterworth']

        # Methode 3: Multivariate Regression
        plot_time_series(data, f'Vorher: {name}', labels='f_x', ylabel='curr_x')
        offset_mv, offsets_mv, mv_scores = mse_random_forest_offset(
            data, target_cols=available_refs, source_cols=source_names, max_shift=max_shift)
        print(f'Offset: {offset_mv}')

        # Nur gemessene Kräfte haben einen zeitlichen versatz
        data = shift_time_series(data, -offset_mv, ['f_x', 'f_y', 'f_z'], 'curr_x')
        data = data.dropna().reset_index(drop=True)

        # Bestimme Bereiche, in denen der Datensatz geteilt werden soll.
        data['materialremoved_sim_butterworth'] = apply_lowpass_filter(data['materialremoved_sim'], cutoff, order)
        nullbereiche = finde_nahe_nullbereiche(data['materialremoved_sim_butterworth'], min_laenge=50)

        # Entferne Randbereiche (falls zu weit außen)
        if nullbereiche:
            if nullbereiche[0]['start'] <= 150:
                print("Erster Bereich zu nah am Start – wird entfernt.")
                nullbereiche.pop(0)
            if nullbereiche and nullbereiche[-1]['ende'] >= (len(data) - 150):
                print("Letzter Bereich zu nah am Ende – wird entfernt.")
                nullbereiche.pop(-1)

        # Ausgabe gefilterter Nullbereiche
        for i, bereich in enumerate(nullbereiche):
            print(f"Übrig Bereich {i + 1}: Start={bereich['start']}, Ende={bereich['ende']}, "
                  f"Länge={bereich['laenge']}, Mitte={bereich['mitte']}")

        # Datensatz aufteilen
        teilbereiche = []
        start_idx = 0
        for bereich in nullbereiche:
            mitte = bereich['mitte']
            teilbereiche.append((start_idx, mitte))
            start_idx = mitte
        # Letzten Abschnitt anhängen (falls noch Daten nach letztem Bereich)
        if start_idx < len(data):
            teilbereiche.append((start_idx, len(data)))

        # Zeitreihen plotten für alle Teilbereiche
        for i, (start, ende) in enumerate(teilbereiche):
            teil = data[['curr_x', 'curr_y', 'curr_z', 'curr_sp',
                         'pos_x', 'pos_y', 'pos_z', 'pos_sp',
                         'a_x', 'a_y', 'a_z', 'a_sp',
                         'v_x', 'v_z', 'v_y', 'v_sp',
                         'f_x', 'f_y', 'f_z',
                         'f_x_sim', 'f_y_sim', 'f_z_sim','f_sp_sim',
                         'materialremoved_sim']].iloc[start:ende].copy()

            if len(teil) > 100:
                plot_time_series(teil, f'Nachher Bereich{i + 1} {name}', labels='f_x', ylabel='curr_x')
                dateiname = f"{name}_{i+1}.csv"
                pfad = os.path.join(output_path, dateiname)
                teil.to_csv(pfad, index=False)
                print(f"Gespeichert: {pfad}")

        print(f'### {name} Ende ###')