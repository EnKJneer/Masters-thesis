import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn

# Beispiel für diskrete Laplace-Transformation
def discrete_laplace_transform(sequence, s_values):
    """
    Berechnet die diskrete Laplace-Transformation
    """
    n = np.arange(len(sequence))
    result = []
    for s in s_values:
        transform = np.sum(sequence * np.exp(-s * n))
        result.append(transform)
    return np.array(result)

def apply_discrete_laplace_transform(df, s_values):
    """
    Wendet die diskrete Laplace-Transformation auf jede Spalte eines DataFrames an.
    """
    return df.apply(lambda column: discrete_laplace_transform(column, s_values))


def analyze_milling_data_xy(X, y=None, dt=0.02):
    """
    Analysiert Fräsdaten (X, y) und empfiehlt s-Werte

    X: Input-Daten (DataFrame oder Array)
    y: Output-Daten (DataFrame oder Array), optional
    dt: Abtastzeit (0.02s = 50 Hz)
    """

    # Konvertiere zu DataFrame falls nötig
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X_df = pd.DataFrame({'signal': X})
        else:
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    else:
        X_df = X.copy()

    if y is not None:
        if isinstance(y, np.ndarray):
            if y.ndim == 1:
                y_df = pd.DataFrame({'target': y})
            else:
                y_df = pd.DataFrame(y, columns=[f'target_{i}' for i in range(y.shape[1])])
        else:
            y_df = y.copy()
    else:
        y_df = None

    print("=== Fräsprozess-Datenanalyse (X/y) ===")
    print(f"X-Datenpunkte: {len(X_df)}")
    print(f"X-Features: {X_df.shape[1]}")
    if y_df is not None:
        print(f"y-Datenpunkte: {len(y_df)}")
        print(f"y-Features: {y_df.shape[1]}")
    print(f"Messdauer: {len(X_df) * dt:.2f} s")

    # Grundlegende Statistiken für X
    print("\n=== X-Daten Statistiken ===")
    for col in X_df.columns:
        data = X_df[col].values
        print(f"{col}:")
        print(f"  RMS: {np.sqrt(np.mean(data ** 2)):.4f}")
        print(f"  Max: {np.max(np.abs(data)):.4f}")
        print(f"  Std: {np.std(data):.4f}")

    # Grundlegende Statistiken für y
    if y_df is not None:
        print("\n=== y-Daten Statistiken ===")
        for col in y_df.columns:
            data = y_df[col].values
            print(f"{col}:")
            print(f"  RMS: {np.sqrt(np.mean(data ** 2)):.4f}")
            print(f"  Max: {np.max(np.abs(data)):.4f}")
            print(f"  Std: {np.std(data):.4f}")

    # s-Werte bestimmen
    s_values_dict = milling_process_s_values_xy(X_df, y_df, dt)

    return s_values_dict, X_df, y_df


def milling_process_s_values_xy(X_df, y_df=None, dt=0.02):
    """
    Bestimmt s-Werte speziell für X/y Fräsprozess-Daten
    """

    f_nyquist = 1 / (2 * dt)  # 25 Hz

    print(f"\nAbtastfrequenz: {1 / dt} Hz")
    print(f"Nyquist-Frequenz: {f_nyquist} Hz")

    # Standard s-Werte für verschiedene Anwendungen
    s_values_dict = {
        'low_frequency': np.linspace(0.1, 20, 50),  # Niederfrequente Prozessdynamik
        'process_dynamics': np.linspace(1, 100, 80),  # Hauptprozessdynamik
        'structural_dynamics': np.linspace(10, 500, 100),  # Strukturdynamik
        'full_range': np.logspace(-1, 3, 150),  # Vollständiger Bereich
        'adaptive_X': None,  # Basierend auf X-Daten
        'adaptive_y': None  # Basierend auf y-Daten
    }

    # Adaptive s-Werte basierend auf X-Daten
    if len(X_df.columns) > 0:
        # Nimm die erste Spalte oder die mit höchster Varianz
        if len(X_df.columns) == 1:
            X_signal = X_df.iloc[:, 0].values
        else:
            # Spalte mit höchster Varianz wählen
            variances = X_df.var()
            max_var_col = variances.idxmax()
            X_signal = X_df[max_var_col].values
            print(f"Verwende X-Spalte '{max_var_col}' für adaptive s-Werte (höchste Varianz)")

        s_values_dict['adaptive_X'] = get_adaptive_s_values(X_signal, dt, f_nyquist)

    # Adaptive s-Werte basierend auf y-Daten
    if y_df is not None and len(y_df.columns) > 0:
        if len(y_df.columns) == 1:
            y_signal = y_df.iloc[:, 0].values
        else:
            # Spalte mit höchster Varianz wählen
            variances = y_df.var()
            max_var_col = variances.idxmax()
            y_signal = y_df[max_var_col].values
            print(f"Verwende y-Spalte '{max_var_col}' für adaptive s-Werte (höchste Varianz)")

        s_values_dict['adaptive_y'] = get_adaptive_s_values(y_signal, dt, f_nyquist)

    return s_values_dict


def get_adaptive_s_values(signal, dt, f_nyquist):
    """
    Bestimmt adaptive s-Werte basierend auf Signalcharakteristik
    """

    # FFT-Analyse für dominante Frequenzen
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), dt)
    power_spectrum = np.abs(fft_signal) ** 2

    # Nur positive Frequenzen
    pos_freqs = freqs[:len(freqs) // 2]
    pos_power = power_spectrum[:len(power_spectrum) // 2]

    # Peak-Frequenzen finden
    try:
        peaks, _ = find_peaks(pos_power, height=np.max(pos_power) * 0.1)
        if len(peaks) > 0:
            dominant_freqs = pos_freqs[peaks]
            f_max = min(np.max(dominant_freqs), f_nyquist)
            print(f"  Dominante Frequenzen gefunden: {dominant_freqs[:5]:.2f} Hz")
        else:
            f_max = f_nyquist / 2
            print(f"  Keine dominanten Frequenzen gefunden, verwende f_max = {f_max:.2f} Hz")
    except:
        f_max = f_nyquist / 2
        print(f"  FFT-Analyse fehlgeschlagen, verwende f_max = {f_max:.2f} Hz")

    # s-Werte bis 3x maximale relevante Frequenz
    s_max_adaptive = 2 * np.pi * f_max * 3
    s_values_adaptive = np.logspace(-1, np.log10(s_max_adaptive), 120)

    return s_values_adaptive


def system_identification_xy(X, y, s_values, dt=0.02):
    """
    Systemidentifikation zwischen X und y mit diskreter Laplace-Transformation

    X: Input-Signal (1D array)
    y: Output-Signal (1D array)
    s_values: s-Werte für DLT
    dt: Abtastzeit
    """

    # Sicherstellen, dass X und y 1D arrays sind
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0].values
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0].values

    if X.ndim > 1:
        X = X.flatten()
    if y.ndim > 1:
        y = y.flatten()

    # Diskrete Laplace-Transformation
    N = min(len(X), len(y))
    X = X[:N]
    y = y[:N]
    n = np.arange(N)

    U_s = []
    Y_s = []

    print(f"Berechne DLT für {N} Datenpunkte und {len(s_values)} s-Werte...")

    for i, s in enumerate(s_values):
        if i % 20 == 0:
            print(f"  Fortschritt: {i}/{len(s_values)}")

        exp_term = np.exp(-s * n * dt)
        u_transform = np.sum(X * exp_term)
        y_transform = np.sum(y * exp_term)
        U_s.append(u_transform)
        Y_s.append(y_transform)

    U_s = np.array(U_s)
    Y_s = np.array(Y_s)

    # Übertragungsfunktion H(s) = Y(s)/U(s)
    # Numerische Stabilität durch kleine Zahl
    H_s = Y_s / (U_s + 1e-10 * np.max(np.abs(U_s)))

    return s_values, H_s, U_s, Y_s


def plot_results_xy(s_values, H_s, U_s, Y_s):
    """
    Visualisiert die Ergebnisse der Systemidentifikation
    """

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Übertragungsfunktion Magnitude
    axes[0, 0].semilogx(s_values, np.abs(H_s))
    axes[0, 0].set_xlabel('s [1/s]')
    axes[0, 0].set_ylabel('|H(s)|')
    axes[0, 0].set_title('Übertragungsfunktion - Magnitude')
    axes[0, 0].grid(True)

    # Übertragungsfunktion Phase
    axes[0, 1].semilogx(s_values, np.angle(H_s) * 180 / np.pi)
    axes[0, 1].set_xlabel('s [1/s]')
    axes[0, 1].set_ylabel('Phase [°]')
    axes[0, 1].set_title('Übertragungsfunktion - Phase')
    axes[0, 1].grid(True)

    # Input-Transformation
    axes[1, 0].semilogx(s_values, np.abs(U_s))
    axes[1, 0].set_xlabel('s [1/s]')
    axes[1, 0].set_ylabel('|U(s)|')
    axes[1, 0].set_title('Input-Transformation')
    axes[1, 0].grid(True)

    # Output-Transformation
    axes[1, 1].semilogx(s_values, np.abs(Y_s))
    axes[1, 1].set_xlabel('s [1/s]')
    axes[1, 1].set_ylabel('|Y(s)|')
    axes[1, 1].set_title('Output-Transformation')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


# Hauptfunktion für Ihre Anwendung
def run_milling_analysis_xy(X_test, y_test, dt=0.02):
    """
    Führt komplette Analyse für alle X/y Paare durch
    """

    results = []

    for i, (X, y) in enumerate(zip(X_test, y_test)):
        print(f"\n{'=' * 50}")
        print(f"Analysiere Datensatz {i + 1}/{len(X_test)}")
        print(f"{'=' * 50}")

        # Datenanalyse
        s_values_dict, X_df, y_df = analyze_milling_data_xy(X, y, dt)

        # Wähle beste s-Werte (adaptive_y falls verfügbar, sonst adaptive_X)
        if s_values_dict['adaptive_y'] is not None:
            s_values = s_values_dict['adaptive_y']
            print(f"\nVerwende adaptive s-Werte basierend auf y-Daten")
        elif s_values_dict['adaptive_X'] is not None:
            s_values = s_values_dict['adaptive_X']
            print(f"\nVerwende adaptive s-Werte basierend auf X-Daten")
        else:
            s_values = s_values_dict['full_range']
            print(f"\nVerwende vollständigen s-Werte Bereich")

        print(f"s-Bereich: {s_values.min():.3f} bis {s_values.max():.3f}")

        # Systemidentifikation
        s_vals, H_s, U_s, Y_s = system_identification_xy(X_df, y_df, s_values, dt)

        # Ergebnisse speichern
        result = {
            'dataset_index': i,
            's_values': s_vals,
            'H_s': H_s,
            'U_s': U_s,
            'Y_s': Y_s,
            'X_df': X_df,
            'y_df': y_df
        }
        results.append(result)

        # Optional: Erste paar Ergebnisse plotten
        if i < 3:  # Nur erste 3 Datensätze plotten
            plot_results_xy(s_vals, H_s, U_s, Y_s)

    return results


if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataClass = hdata.Combined_Plate_TrainVal
    dataClass.target_channels = ['curr_x']
    dataClass.past_values = past_values
    dataClass.future_values = future_values
    dataClass.window_size = window_size

    dataClasses = [dataClass] #, hdata.Combined_Plate_TrainVal_CONTDEV

    #model_simple = mphys.NaiveModelSimple()
    model = mnn.get_reference()
    models = [model]

    # Ihre Daten laden
    X_train, X_val, X_test, y_train, y_val, y_test = dataClass.load_data()

    # Oder für einzelne Datensätze:
    for i, (X, y) in enumerate(zip(X_test[:3], y_test[:3])):  # Nur erste 3
        n = X[X['materialremoved_sim_1_current'] > 0].index.min()
        X = X.iloc[:n, :]
        y = y.iloc[:n, :]
        print(f"\n=== Datensatz {i + 1} ===")
        s_values_dict, X_df, y_df = analyze_milling_data_xy(X, y)

        # Systemidentifikation mit besten s-Werten
        if s_values_dict['adaptive_y'] is not None:
            s_values = s_values_dict['adaptive_y']
        else:
            s_values = s_values_dict['adaptive_X']

        s_vals, H_s, U_s, Y_s = system_identification_xy(X_df, y_df, s_values)

        # Ergebnisse anzeigen
        print(f"Übertragungsfunktion |H(s)| max: {np.max(np.abs(H_s)):.4f}")
        print(f"Übertragungsfunktion |H(s)| min: {np.min(np.abs(H_s)):.4f}")