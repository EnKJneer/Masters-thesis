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

def cross_correlation_offset(signal1, signal2, max_shift=500):
    """
    Bestimmt den Offset mittels Kreuzkorrelation.
    Positiver Offset: signal2 startet später (muss nach links verschoben werden)
    Negativer Offset: signal2 startet früher (muss nach rechts verschoben werden)
    """
    # Entferne NaN-Werte
    valid_idx1 = ~np.isnan(signal1)
    valid_idx2 = ~np.isnan(signal2)

    if np.sum(valid_idx1) < 100 or np.sum(valid_idx2) < 100:
        return 0, np.array([])

    s1 = signal1[valid_idx1]
    s2 = signal2[valid_idx2]

    # Normalisiere die Signale
    s1_norm = (s1 - np.mean(s1)) / (np.std(s1) + 1e-10)
    s2_norm = (s2 - np.mean(s2)) / (np.std(s2) + 1e-10)

    # Begrenze max_shift auf sinnvolle Werte
    max_shift = min(max_shift, len(s1) // 3, len(s2) // 3)

    # Berechne Kreuzkorrelation für verschiedene Shifts
    correlations = []
    shifts = range(-max_shift, max_shift + 1)

    for shift in shifts:
        if shift == 0:
            # Kein Shift - verwende den Überlappungsbereich
            min_len = min(len(s1_norm), len(s2_norm))
            corr = np.corrcoef(s1_norm[:min_len], s2_norm[:min_len])[0, 1]
        elif shift > 0:
            # s2 startet später - s1 von Position shift, s2 von Anfang
            if shift >= len(s1_norm):
                corr = 0
            else:
                s1_part = s1_norm[shift:]
                s2_part = s2_norm[:len(s1_part)]
                min_len = min(len(s1_part), len(s2_part))
                if min_len > 0:
                    corr = np.corrcoef(s1_part[:min_len], s2_part[:min_len])[0, 1]
                else:
                    corr = 0
        else:  # shift < 0
            # s2 startet früher - s1 von Anfang, s2 von Position -shift
            if -shift >= len(s2_norm):
                corr = 0
            else:
                s1_part = s1_norm[:len(s1_norm) + shift]
                s2_part = s2_norm[-shift:]
                min_len = min(len(s1_part), len(s2_part))
                if min_len > 0:
                    corr = np.corrcoef(s1_part[:min_len], s2_part[:min_len])[0, 1]
                else:
                    corr = 0

        correlations.append(corr if not np.isnan(corr) else 0)

    correlations = np.array(correlations)

    # Finde das Maximum
    max_idx = np.argmax(correlations)
    best_offset = shifts[max_idx]

    return best_offset, correlations

def mse_single_signal_offset(signal1, signal2, max_shift=500):
    """
    MSE-Optimierung für ein einzelnes Signal.
    """
    valid_idx1 = ~np.isnan(signal1)
    valid_idx2 = ~np.isnan(signal2)

    if np.sum(valid_idx1) < 100 or np.sum(valid_idx2) < 100:
        return 0, [], []

    s1 = signal1[valid_idx1]
    s2 = signal2[valid_idx2]

    max_shift = min(max_shift, len(s1) // 3, len(s2) // 3)

    def calculate_mse(offset):
        offset = int(offset)
        if offset == 0:
            min_len = min(len(s1), len(s2))
            return mean_squared_error(s1[:min_len], s2[:min_len])
        elif offset > 0:
            # s2 startet später
            if offset >= len(s1):
                return np.inf
            s1_part = s1[offset:]
            s2_part = s2[:len(s1_part)]
            min_len = min(len(s1_part), len(s2_part))
            if min_len == 0:
                return np.inf
            return mean_squared_error(s1_part[:min_len], s2_part[:min_len])
        else:  # offset < 0
            # s2 startet früher
            if -offset >= len(s2):
                return np.inf
            s1_part = s1[:len(s1) + offset]
            s2_part = s2[-offset:]
            min_len = min(len(s1_part), len(s2_part))
            if min_len == 0:
                return np.inf
            return mean_squared_error(s1_part[:min_len], s2_part[:min_len])

    # Optimiere den Offset
    #result = minimize_scalar(calculate_mse, bounds=(-max_shift, max_shift), method='bounded')
    #best_offset = int(result.x)

    # Berechne MSE-Werte für Visualisierung
    #offsets = range(-max_shift, max_shift + 1, 20)
    #mse_values = [calculate_mse(offset) for offset in offsets]

    offsets = range(-max_shift, max_shift + 1, 20)
    scores = [calculate_mse(offset) for offset in offsets]
    best_offset = offsets[scores.index(min(scores))]

    offsets_highres = range(best_offset - 11, best_offset + 11, 1)
    scores_highres = [calculate_mse(offset) for offset in offsets_highres]
    best_offset = offsets_highres[scores_highres.index(min(scores_highres))]

    return best_offset, offsets, scores

def mse_multivariate_regression_offset(data, target_cols=['curr_x', 'curr_y', 'curr_z', 'curr_sp','a_x', 'a_y', 'a_z', 'a_sp', 'v_x', 'v_y', 'v_z', 'v_sp'], source_col=['f_x_butterworth', 'f_y_butterworth', 'f_z_butterworth'], max_shift=500):
    """
    MSE-Optimierung mit multivariater linearer Regression.
    Verwendet mehrere Signale zur Bestimmung des optimalen Offsets.

    Parameters:
    data: DataFrame mit den Signalen
    target_cols: Liste der Referenzsignale
    source_col: Zu verschiebendes Signal
    max_shift: Maximaler Verschiebungsbereich
    """
    # Prüfe welche Spalten verfügbar sind
    available_cols = [col for col in target_cols if col in data.columns]
    #if not available_cols or source_col not in data.columns:
    #    print(f"Warnung: Nicht alle Spalten verfügbar. Verfügbar: {data.columns.tolist()}")
    #    return 0, [], []

    # Erstelle saubere Daten ohne NaN
    clean_data = data[available_cols + [source_col]].dropna()

    if len(clean_data) < 100:
        print("Warnung: Zu wenige saubere Datenpunkte für multivariate Regression")
        return 0, [], []

    X_target = clean_data[available_cols].values
    y_source = clean_data[source_col].values

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
            reg = RandomForestRegressor(n_estimators=10)
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

def visualize_offset_methods(data, max_shift_seconds=5, sampling_rate=50):
    """
    Visualisiert die verschiedenen Methoden zur Offset-Bestimmung.

    Parameters:
    data: DataFrame mit den Signalen
    max_shift_seconds: Maximaler Verschiebungsbereich in Sekunden
    sampling_rate: Abtastrate in Hz
    """
    max_shift = int(max_shift_seconds * sampling_rate)

    # Bestimme verfügbare Referenzsignale
    ref_signals = ['curr_x', 'a_x', 'a_y', 'a_z']
    available_refs = [col for col in ref_signals if col in data.columns]

    if not available_refs:
        print("Fehler: Keine Referenzsignale gefunden")
        return {}

    # Verwende das erste verfügbare Referenzsignal für einfache Methoden
    ref_signal = data[available_refs[0]].values

    # Bestimme das zu verschiebende Signal
    if 'f_x_butterworth' in data.columns:
        source_signal = data['f_x_butterworth'].values
        source_name = 'f_x_butterworth'
    elif 'f_x' in data.columns:
        source_signal = data['f_x'].values
        source_name = 'f_x'
    else:
        print("Fehler: Kein f_x Signal gefunden")
        return {}

    print(f"Verwende {available_refs[0]} als Referenz und {source_name} als Quellsignal")
    print(f"Maximaler Verschiebungsbereich: ±{max_shift_seconds}s (±{max_shift} Samples)")

    # Methode 1: Kreuzkorrelation
    offset_cc, correlation = cross_correlation_offset(ref_signal, source_signal, max_shift)

    # Methode 2: MSE-Optimierung (einzelnes Signal)
    offset_mse, offsets_mse, mse_values = mse_single_signal_offset(ref_signal, source_signal, max_shift)

    # Methode 3: Multivariate Regression
    offset_mv, offsets_mv, mv_scores = mse_multivariate_regression_offset(
        data, target_cols=available_refs, source_col=source_name, max_shift=max_shift)

    # Visualisierung
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Ursprüngliche Signale
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()

    ax1.plot(ref_signal, label=f'{available_refs[0]} (Referenz)', color='blue', alpha=0.7)
    ax1_twin.plot(source_signal, label=f'{source_name} (zu verschieben)', color='orange', alpha=0.7)

    ax1.set_xlabel('Index')
    ax1.set_ylabel(f'{available_refs[0]}', color='blue')
    ax1_twin.set_ylabel(f'{source_name}', color='orange')
    ax1.set_title('Ursprüngliche Signale')
    ax1.grid(True, alpha=0.3)

    # Kombiniere Legenden
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Plot 2: Kreuzkorrelation
    if len(correlation) > 0:
        shifts = np.arange(-max_shift, max_shift + 1)
        if len(shifts) == len(correlation):
            axes[0, 1].plot(shifts / sampling_rate, correlation)
            axes[0, 1].axvline(x=offset_cc / sampling_rate, color='red', linestyle='--',
                               label=f'Offset: {offset_cc / sampling_rate:.2f}s')
            axes[0, 1].set_title('Kreuzkorrelation')
            axes[0, 1].set_xlabel('Offset (Sekunden)')
            axes[0, 1].set_ylabel('Korrelation')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: MSE-Vergleich
    ax3 = axes[1, 0]
    if len(mse_values) > 0:
        finite_mse = [mse if np.isfinite(mse) else np.nan for mse in mse_values]
        ax3.plot(np.array(offsets_mse) / sampling_rate, finite_mse, 'b-',
                 label=f'MSE Einzelsignal (Offset: {offset_mse / sampling_rate:.2f}s)')
        ax3.axvline(x=offset_mse / sampling_rate, color='blue', linestyle='--', alpha=0.7)

    if len(mv_scores) > 0:
        finite_mv = [score if np.isfinite(score) else np.nan for score in mv_scores]
        ax3_twin = ax3.twinx()
        ax3_twin.plot(np.array(offsets_mv) / sampling_rate, finite_mv, 'g-',
                      label=f'MSE Regression (Offset: {offset_mv / sampling_rate:.2f}s)')
        ax3_twin.axvline(x=offset_mv / sampling_rate, color='green', linestyle='--', alpha=0.7)
        ax3_twin.set_ylabel('Regression Score', color='green')

    ax3.set_title('MSE-Optimierung Vergleich')
    ax3.set_xlabel('Offset (Sekunden)')
    ax3.set_ylabel('MSE Einzelsignal', color='blue')
    ax3.legend(loc='upper left')
    if len(mv_scores) > 0:
        ax3_twin.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Vergleich der Methoden
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    ax4.plot(ref_signal, label=f'{available_refs[0]} (Referenz)', color='blue', alpha=0.7)

    methods = [
        (offset_cc, 'Kreuzkorrelation', 'red'),
        (offset_mse, 'MSE Einzelsignal', 'green'),
        (offset_mv, 'MSE Regression', 'purple')
    ]

    for offset, method, color in methods:
        shifted_signal = apply_offset_to_signal(source_signal, offset)
        # Beschränke auf ursprüngliche Länge
        if len(shifted_signal) > len(ref_signal):
            shifted_signal = shifted_signal[:len(ref_signal)]
        elif len(shifted_signal) < len(ref_signal):
            shifted_signal = np.concatenate([shifted_signal,
                                             np.full(len(ref_signal) - len(shifted_signal), np.nan)])

        ax4_twin.plot(shifted_signal, label=f'{method} ({offset / sampling_rate:.2f}s)',
                      alpha=0.6, color=color)

    ax4.set_xlabel('Index')
    ax4.set_ylabel(f'{available_refs[0]}', color='blue')
    ax4_twin.set_ylabel(f'{source_name}', color='orange')
    ax4.set_title('Vergleich der Methoden')
    ax4.grid(True, alpha=0.3)

    # Kombiniere Legenden
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()

    # Ausgabe der Ergebnisse
    print("\n" + "=" * 50)
    print("ERGEBNISSE DER OFFSET-BESTIMMUNG")
    print("=" * 50)
    print(f"Kreuzkorrelation:     {offset_cc:4d} Samples ({offset_cc / sampling_rate:+6.2f}s)")
    print(f"MSE Einzelsignal:     {offset_mse:4d} Samples ({offset_mse / sampling_rate:+6.2f}s)")
    print(f"MSE Regression:       {offset_mv:4d} Samples ({offset_mv / sampling_rate:+6.2f}s)")

    print(f"\nVerfügbare Referenzsignale: {available_refs}")
    print(f"Quellsignal: {source_name}")
    print(f"Abtastrate: {sampling_rate} Hz")
    print(f"Maximaler Suchbereich: ±{max_shift_seconds}s")

    return {
        'cross_correlation': offset_cc,
        'mse_single': offset_mse,
        'mse_regression': offset_mv
    }

path_data = 'Old_CombinedData'
files = os.listdir(path_data)
files = ['AL_2007_T4_Plate_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']
files = ['S235JR_Plate_Normal.csv']
for file in files:
    name = file.replace('.csv', '')
    data = pd.read_csv(f'{path_data}/{file}')
    #n = int(len(data)/3)
    #data = data.iloc[:n, :]
    data['f_x'] = data['f_x'].apply(lambda x: 0 if x > 1000 else x)
    # Berechnung der Standardabweichung
    std_dev = data['f_x'].std()
    # Berechnung des Mittelwerts
    mean = data['f_x'].mean()
    # Festlegung der Grenzen basierend auf der Standardabweichung
    lower_bound = mean - 4 * std_dev
    upper_bound = mean + 4 * std_dev
    # Begrenzen der Daten
    data['f_x'] = np.clip(data['f_x'], lower_bound, upper_bound)
    # Butterworth-Filter-Parameter
    cutoff = 0.1  # Normalisierte Grenzfrequenz (z.B. 0.1 bedeutet 10% der Nyquist-Frequenz)
    order = 4  # Filterordnung
    def butter_lowpass(cutoff, order, nyq_freq=0.5):
        normal_cutoff = cutoff / nyq_freq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    def apply_lowpass_filter(data, cutoff, order):
        b, a = butter_lowpass(cutoff, order)
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    # Butterworth-Filter auf die Daten anwenden
    axes = ['x', 'y', 'z']
    for axis in axes:
        data[f'f_{axis}_butterworth'] = apply_lowpass_filter(data[f'f_{axis}'], cutoff, order)
    # Gleitender Mittelwertfilter
    def moving_average(data, window_size):
        return data.rolling(window=window_size, center=True).mean()
    window_size = 5  # Fenstergröße für den gleitenden Mittelwertfilter
    data['f_x_moving_avg'] = moving_average(data['f_x'], window_size)

    offsets = visualize_offset_methods(data)

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

    i = int(len(data)/3)
    data = data.iloc[:i, :]

    # Annahme: 'data' ist dein DataFrame und enthält die Spalten 'f_x_butterworth', 'f_x_moving_avg', und 'curr_x'
    for key, n in offsets.items():
        data_2 = shift_time_series(data, -n, ['f_x_butterworth', 'f_x_moving_avg'], 'curr_x')
        plot_time_series(data_2, f'Nachher: {name} ({key})', labels='f_x_butterworth', ylabel='curr_x')
        plot_time_series(data_2, f'Nachher: {name} ({key})', labels='materialremoved_sim', ylabel='curr_x')