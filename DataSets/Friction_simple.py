import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

def sign_hold(v_x):
    signs = np.sign(v_x)
    result = signs.copy()

    # Iteriere über jedes Element im Array
    for i in range(len(signs)):
        if signs[i] == 0:
            # Sammle die letzten fünf nicht-null Vorzeichen
            last_five_non_zero = []
            j = i - 1
            while j >= 0 and len(last_five_non_zero) < 5:
                if signs[j] != 0:
                    last_five_non_zero.append(signs[j])
                j -= 1

            # Berechne die Summe der letzten fünf nicht-null Vorzeichen
            if last_five_non_zero:
                sum_signs = np.sum(last_five_non_zero)
                # Bestimme das Vorzeichen der Summe
                result[i] = np.sign(sum_signs)

    return result

def fit_friction_model(data, velocity_threshold=1e-6, acceleration_threshold=1e-6):
    """
    Fittet das zweistufige Reibungsmodell:
    - Stillstand (v_x ≈ 0, a_x ≈ 0): y = F_s * sign_hold(v_x) + a_s * f_x_sim + b_s
    - Bewegung: y = F_c * sign(v_x) + sigma_2 * v_x + a_d * f_x_sim + b_d
    """
    # Extrahiere relevante Variablen
    v_x = data['v_x'].values
    v_s = sign_hold(v_x)
    a_x = data['a_x'].values
    f_x_sim = data['f_x_sim'].values
    curr_x = data['curr_x'].values

    # Definiere Stillstands- und Bewegungsmasken
    stillstand_mask = (np.abs(v_x) <= velocity_threshold) & (np.abs(a_x) <= acceleration_threshold)
    bewegung_mask = ~stillstand_mask

    print(f"Anzahl Stillstandspunkte: {np.sum(stillstand_mask)}")
    print(f"Anzahl Bewegungspunkte: {np.sum(bewegung_mask)}")

    # Parameter-Dictionary für Ergebnisse
    params = {}

    # === Stillstandsmodell fitten ===
    if np.sum(stillstand_mask) > 2:
        # y = F_s * sign_hold(v_x) + a_s * f_x_sim + b_s
        X_stillstand = np.column_stack([
            v_s[stillstand_mask],
            f_x_sim[stillstand_mask],
            np.ones(np.sum(stillstand_mask))  # Bias term
        ])
        y_stillstand = curr_x[stillstand_mask]

        reg_stillstand = LinearRegression(fit_intercept=False)
        reg_stillstand.fit(X_stillstand, y_stillstand)

        params['F_s'] = reg_stillstand.coef_[0]
        params['a_s'] = reg_stillstand.coef_[1]
        params['b_s'] = reg_stillstand.coef_[2]
    else:
        print("Warnung: Nicht genügend Stillstandspunkte für Fitting")
        params['F_s'] = 0
        params['a_s'] = 1
        params['b_s'] = 0

    # === Bewegungsmodell fitten ===
    if np.sum(bewegung_mask) > 3:
        # y = F_c * sign(v_x) + sigma_2 * v_x + a_d * f_x_sim + b_d
        X_bewegung = np.column_stack([
            np.sign(v_x[bewegung_mask]),
            v_x[bewegung_mask],
            f_x_sim[bewegung_mask],
            np.ones(np.sum(bewegung_mask))  # Bias term
        ])
        y_bewegung = curr_x[bewegung_mask]

        reg_bewegung = LinearRegression(fit_intercept=False)
        reg_bewegung.fit(X_bewegung, y_bewegung)

        params['F_c'] = reg_bewegung.coef_[0]
        params['sigma_2'] = reg_bewegung.coef_[1]
        params['a_d'] = reg_bewegung.coef_[2]
        params['b_d'] = reg_bewegung.coef_[3]
    else:
        print("Warnung: Nicht genügend Bewegungspunkte für Fitting")
        params['F_c'] = 0
        params['sigma_2'] = 0
        params['a_d'] = 1
        params['b_d'] = 0

    return params, stillstand_mask, bewegung_mask

def predict_model(data, params, stillstand_mask, bewegung_mask):
    """Vorhersage mit dem gefitteten Modell"""
    v_x = data['v_x'].values
    v_s = sign_hold(v_x)
    f_x_sim = data['f_x_sim'].values
    y_pred = np.zeros_like(v_x)

    # Stillstandsvorhersage
    if np.sum(stillstand_mask) > 0:
        y_pred[stillstand_mask] = (params['F_s'] * v_s[stillstand_mask] +
                                   params['a_s'] * f_x_sim[stillstand_mask] +
                                   params['b_s'])


    # Bewegungsvorhersage
    if np.sum(bewegung_mask) > 0:
        y_pred[bewegung_mask] = (params['F_c'] * np.sign(v_x[bewegung_mask]) +
                                 params['sigma_2'] * v_x[bewegung_mask] +
                                 params['a_d'] * f_x_sim[bewegung_mask] +
                                 params['b_d'])

    return y_pred

def plot_results(data, y_pred, mae, file_name, params, stillstand_mask, bewegung_mask):
    """Plotte Zeitreihe mit MAE und Modellbereichen"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    time_index = np.arange(len(data))

    # Plot 1: Gemessene vs vorhergesagte Werte
    ax1.plot(time_index, data['curr_x'], 'b-', label='Gemessen', alpha=0.7)
    ax1.plot(time_index, y_pred, 'r--', label='Vorhergesagt', alpha=0.7)
    ax1.set_ylabel('Strom X [A]')
    ax1.set_title(f'{file_name} - Modellfit (MAE: {mae:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Aktive Modellbereiche
    modell_bereiche = np.zeros_like(time_index, dtype=float)
    modell_bereiche[stillstand_mask] = 1  # Stillstandsmodell
    modell_bereiche[bewegung_mask] = 2  # Bewegungsmodell

    ax2.fill_between(time_index, 0, modell_bereiche,
                     where=(modell_bereiche == 1), color='orange', alpha=0.6,
                     label='Stillstandsmodell', step='pre')
    ax2.fill_between(time_index, 0, modell_bereiche,
                     where=(modell_bereiche == 2), color='green', alpha=0.6,
                     label='Bewegungsmodell', step='pre')
    ax2.set_ylabel('Aktives Modell')
    ax2.set_ylim(0, 2.5)
    ax2.set_yticks([0.5, 1.5])
    ax2.set_yticklabels(['Stillstand', 'Bewegung'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuen
    residuals = data['curr_x'] - y_pred
    ax3.plot(time_index, residuals, 'k-', alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Residuen [A]')
    ax3.set_xlabel('Zeitindex')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def filter_constant_z(data, z_tolerance=0.1):
    """Filtert Daten für konstantes pos_z"""
    z_std = data['pos_z'].std()
    if z_std <= z_tolerance:
        return data
    else:
        # Finde den häufigsten Z-Wert
        z_mode = data['pos_z'].mode().iloc[0]
        mask = np.abs(data['pos_z'] - z_mode) <= z_tolerance
        print(f"Filtere für konstantes Z: {np.sum(mask)}/{len(data)} Punkte behalten")
        return data[mask].reset_index(drop=True)


# === Main ===
if __name__ == '__main__':
    path_data = 'DataFiltered'
    files = ['AL_2007_T4_Plate_Normal_3.csv'],

    """             'AL_2007_T4_Gear_Normal_3.csv',
             'S235JR_Gear_Normal_3.csv',
             'S235JR_Plate_Normal_3.csv']"""

    results_summary = []

    for file in files:
        # Sicherheitscheck falls file eine Liste ist
        if isinstance(file, list):
            print(f"Warnung: file ist eine Liste: {file}. Nehme erstes Element.")
            file = file[0]

        print(f'\n{"=" * 50}')
        print(f'Verarbeite Datei: {file}')
        print(f'{"=" * 50}')

        try:
            # Lade Daten (file ist bereits ein String, nicht eine Liste)
            file_path = os.path.join(path_data, file)
            data = pd.read_csv(file_path)
            print(f"Ursprüngliche Datenpunkte: {len(data)}")

            # Filtere für konstantes pos_z
            data_filtered = filter_constant_z(data)

            data_filtered = data_filtered.iloc[:-160]
            data_filtered = data_filtered.iloc[160:]
            data_filtered = data_filtered.reset_index(drop=True)

            plt.plot(data_filtered.index, data_filtered['pos_z'])
            plt.show()
            # Erstellen des Plots
            fig, ax1 = plt.subplots()

            # Plot auf der ersten y-Achse
            ax1.plot(data_filtered.index, np.sign(data_filtered['v_x']), 'b-', label='v_x')
            #ax1.plot(data_filtered.index, sign_hold(data_filtered['v_x']), 'g-', label='sign_hold(v_x)')
            #ax1.plot(data_filtered.index, (data_filtered['f_x_sim']) / 200, 'r-', label='f_x_sim')
            ax1.set_xlabel('Index')
            ax1.set_ylabel('Werte')

            # Hinzufügen einer zweiten y-Achse
            ax2 = ax1.twinx()
            ax2.plot(data_filtered.index, -data_filtered['curr_x'], 'm-', label='curr_x')
            ax2.set_ylabel('curr_y Werte')

            # Legende hinzufügen
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            plt.show()


            print(f"Nach Z-Filterung: {len(data_filtered)}")

            if len(data_filtered) < 10:
                print("Nicht genügend Datenpunkte nach Filterung. Überspringe Datei.")
                continue

            # Fitte Modell
            params, stillstand_mask, bewegung_mask = fit_friction_model(data_filtered)

            # Vorhersage
            y_pred = predict_model(data_filtered, params, stillstand_mask, bewegung_mask)

            # Berechne MAE
            mae = mean_absolute_error(data_filtered['curr_x'], y_pred)

            # Ausgabe der Parameter
            print(f"\nGefittete Parameter für {file}:")
            print(f"Stillstandsreibung (F_s): {params['F_s']:.6f}")
            print(f"Coulomb-Reibung (F_c): {params['F_c']:.6f}")
            print(f"Viskose Dämpfung (sigma_2): {params['sigma_2']:.6f}")
            print(f"Bias Stillstand (b_s): {params['b_s']:.6f}")
            print(f"Bias Bewegung (b_d): {params['b_d']:.6f}")
            print(f"Sim-Kraft Koeff. Stillstand (a_s): {params['a_s']:.6f}")
            print(f"Sim-Kraft Koeff. Bewegung (a_d): {params['a_d']:.6f}")
            print(f"MAE: {mae:.6f}")

            # Plotte Ergebnisse
            plot_results(data_filtered, y_pred, mae, file, params, stillstand_mask, bewegung_mask)

            # Sammle Ergebnisse
            result = {'file': file, 'mae': mae}
            result.update(params)
            results_summary.append(result)

        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {file}: {e}")
            continue

    # Zusammenfassung
    print(f'\n{"=" * 60}')
    print("ZUSAMMENFASSUNG ALLER DATEIEN:")
    print(f'{"=" * 60}')

    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        for _, row in summary_df.iterrows():
            print(f"\n{row['file']}:")
            print(f"  MAE: {row['mae']:.6f}")
            print(f"  F_s: {row['F_s']:.6f}, F_c: {row['F_c']:.6f}")
            print(f"  sigma_2: {row['sigma_2']:.6f}")
            print(f"  a_s: {row['a_s']:.6f}, a_d: {row['a_d']:.6f}")

        print(f"\nDurchschnittliche MAE: {summary_df['mae'].mean():.6f}")
    else:
        print("Keine erfolgreichen Fits.")