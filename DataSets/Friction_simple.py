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
    - Bewegung: y = F_c * sign(v_x) + sigma_2 * v_x + a_d * f_x_sim + a_b * a_x + b_d
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
    if np.sum(bewegung_mask) > 4:  # Jetzt 5 Parameter, also mindestens 5 Punkte
        # y = F_c * sign(v_x) + sigma_2 * v_x + a_d * f_x_sim + a_b * a_x + b_d
        X_bewegung = np.column_stack([
            np.sign(v_x[bewegung_mask]),
            v_x[bewegung_mask],
            f_x_sim[bewegung_mask],
            a_x[bewegung_mask],  # Neuer Beschleunigungsterm
            np.ones(np.sum(bewegung_mask))  # Bias term
        ])
        y_bewegung = curr_x[bewegung_mask]

        reg_bewegung = LinearRegression(fit_intercept=False)
        reg_bewegung.fit(X_bewegung, y_bewegung)

        params['F_c'] = reg_bewegung.coef_[0]
        params['sigma_2'] = reg_bewegung.coef_[1]
        params['a_d'] = reg_bewegung.coef_[2]
        params['a_b'] = reg_bewegung.coef_[3]  # Beschleunigungskoeffizient
        params['b_d'] = reg_bewegung.coef_[4]
    else:
        print("Warnung: Nicht genügend Bewegungspunkte für Fitting")
        params['F_c'] = 0
        params['sigma_2'] = 0
        params['a_d'] = 1
        params['a_b'] = 0
        params['b_d'] = 0

    return params, stillstand_mask, bewegung_mask


def predict_model(data, params, stillstand_mask=None, bewegung_mask=None):
    """Vorhersage mit dem gefitteten Modell"""
    v_x = data['v_x'].values
    v_s = sign_hold(v_x)
    a_x = data['a_x'].values
    f_x_sim = data['f_x_sim'].values
    y_pred = np.zeros_like(v_x)

    # Wenn keine Masken übergeben werden, berechne sie neu
    if stillstand_mask is None or bewegung_mask is None:
        stillstand_mask = (np.abs(v_x) <= 1e-6) & (np.abs(a_x) <= 1e-6)
        bewegung_mask = ~stillstand_mask

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
                                 params['a_b'] * a_x[bewegung_mask] +  # Neuer Term
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


def preprocess_data(data):
    """Vorverarbeitung der Daten wie in der ursprünglichen Version"""
    # Filtere für konstantes pos_z
    data_filtered = filter_constant_z(data)

    # Entferne Anfang und Ende wie im Original
    data_filtered = data_filtered.iloc[:-160]
    data_filtered = data_filtered.iloc[160:]
    data_filtered = data_filtered.reset_index(drop=True)

    return data_filtered


# === Main ===
if __name__ == '__main__':
    path_data = 'DataFiltered'

    # Training-Datei (zum Fitten des Modells)
    training_file = 'AL_2007_T4_Plate_Normal_3.csv'

    # Test-Dateien (zum Testen des gefitteten Modells)
    test_files = ['AL_2007_T4_Plate_Normal_3.csv',
                  'AL_2007_T4_Gear_Normal_3.csv',
                  'S235JR_Gear_Normal_3.csv',
                  'S235JR_Plate_Normal_3.csv']

    print(f'{"=" * 60}')
    print("SCHRITT 1: MODELL TRAINING")
    print(f'{"=" * 60}')
    print(f'Trainiere Modell auf: {training_file}')

    # === TRAINING PHASE ===
    try:
        # Lade Training-Daten
        training_data = pd.read_csv(os.path.join(path_data, training_file))
        print(f"Ursprüngliche Trainingspunkte: {len(training_data)}")

        # Vorverarbeitung
        training_data_processed = preprocess_data(training_data)
        print(f"Nach Preprocessing: {len(training_data_processed)}")

        if len(training_data_processed) < 10:
            print("Nicht genügend Trainingsdaten. Abbruch.")
            exit()

        # Fitte Modell auf Trainingsdaten
        trained_params, train_stillstand_mask, train_bewegung_mask = fit_friction_model(training_data_processed)

        # Trainings-Performance
        train_pred = predict_model(training_data_processed, trained_params, train_stillstand_mask, train_bewegung_mask)
        train_mae = mean_absolute_error(training_data_processed['curr_x'], train_pred)

        print(f"\nTrainierte Parameter:")
        print(f"Stillstandsreibung (F_s): {trained_params['F_s']:.6f}")
        print(f"Coulomb-Reibung (F_c): {trained_params['F_c']:.6f}")
        print(f"Viskose Dämpfung (sigma_2): {trained_params['sigma_2']:.6f}")
        print(f"Beschleunigungskoeff. (a_b): {trained_params['a_b']:.6f}")
        print(f"Bias Stillstand (b_s): {trained_params['b_s']:.6f}")
        print(f"Bias Bewegung (b_d): {trained_params['b_d']:.6f}")
        print(f"Sim-Kraft Koeff. Stillstand (a_s): {trained_params['a_s']:.6f}")
        print(f"Sim-Kraft Koeff. Bewegung (a_d): {trained_params['a_d']:.6f}")
        print(f"Training MAE: {train_mae:.6f}")

        # Plot Training-Ergebnisse
        plot_results(training_data_processed, train_pred, train_mae, f"{training_file} (Training)",
                     trained_params, train_stillstand_mask, train_bewegung_mask)

    except Exception as e:
        print(f"Fehler beim Training: {e}")
        exit()

    print(f'\n{"=" * 60}')
    print("SCHRITT 2: MODELL TESTING")
    print(f'{"=" * 60}')

    # === TESTING PHASE ===
    test_results = []

    for test_file in test_files:
        print(f'\nTeste auf: {test_file}')

        try:
            # Lade Test-Daten
            test_data = pd.read_csv(os.path.join(path_data, test_file))
            print(f"Ursprüngliche Testpunkte: {len(test_data)}")

            # Vorverarbeitung
            test_data_processed = preprocess_data(test_data)
            print(f"Nach Preprocessing: {len(test_data_processed)}")

            if len(test_data_processed) < 10:
                print("Nicht genügend Testdaten. Überspringe.")
                continue

            # Verwende trainierte Parameter für Vorhersage
            test_pred = predict_model(test_data_processed, trained_params)
            test_mae = mean_absolute_error(test_data_processed['curr_x'], test_pred)

            print(f"Test MAE: {test_mae:.6f}")

            # Bestimme Masken für Plotting
            v_x = test_data_processed['v_x'].values
            a_x = test_data_processed['a_x'].values
            test_stillstand_mask = (np.abs(v_x) <= 1e-6) & (np.abs(a_x) <= 1e-6)
            test_bewegung_mask = ~test_stillstand_mask

            # Plot Test-Ergebnisse
            plot_results(test_data_processed, test_pred, test_mae, f"{test_file} (Test)",
                         trained_params, test_stillstand_mask, test_bewegung_mask)

            # Sammle Ergebnisse
            test_results.append({
                'file': test_file,
                'mae': test_mae,
                'is_training_file': test_file == training_file
            })

        except Exception as e:
            print(f"Fehler beim Testen von {test_file}: {e}")
            continue

    # === ZUSAMMENFASSUNG ===
    print(f'\n{"=" * 80}')
    print("GESAMTZUSAMMENFASSUNG")
    print(f'{"=" * 80}')

    print(f"Trainiert auf: {training_file}")
    print(f"Training MAE: {train_mae:.6f}")
    print(f"\nTest-Ergebnisse:")

    for result in test_results:
        status = "(Training-Datei)" if result['is_training_file'] else "(Test-Datei)"
        print(f"  {result['file']} {status}: MAE = {result['mae']:.6f}")

    # Durchschnittliche Performance auf reinen Test-Dateien
    pure_test_results = [r for r in test_results if not r['is_training_file']]
    if pure_test_results:
        avg_test_mae = np.mean([r['mae'] for r in pure_test_results])
        print(f"\nDurchschnittliche MAE auf reinen Test-Dateien: {avg_test_mae:.6f}")

    print(f"\nErweiterte Modellgleichungen:")
    print(f"Stillstand: y = F_s * sign_hold(v_x) + a_s * f_x_sim + b_s")
    print(f"Bewegung:   y = F_c * sign(v_x) + sigma_2 * v_x + a_d * f_x_sim + a_b * a_x + b_d")