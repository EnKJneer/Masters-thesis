import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def find_matching_sequence(data, data_reduced, sequence_length=10):
    """
    Findet die Position von data (Ausschnitt) in data_reduced (vollständige Daten)
    basierend auf Sequenz-Matching der Positions- und Stromdaten
    """
    # Verwende die ersten sequence_length Werte aus data als Suchpattern
    pattern_pos = data[['pos_x', 'pos_y', 'pos_z']].iloc[:sequence_length].values
    pattern_curr = data[['curr_x', 'curr_y', 'curr_z', 'curr_sp']].iloc[:sequence_length].values

    best_match_idx = -1
    best_score = float('inf')

    # Durchsuche data_reduced nach der besten Übereinstimmung
    for start_idx in range(len(data_reduced) - len(data) + 1):
        # Extrahiere Vergleichssequenz aus data_reduced
        compare_pos = data_reduced[['ENC_POS_X', 'ENC_POS_Y', 'ENC_POS_Z']].iloc[
                      start_idx:start_idx + sequence_length].values
        compare_curr = data_reduced[['CURRENT_X', 'CURRENT_Y', 'CURRENT_Z', 'CURRENT_SP']].iloc[
                       start_idx:start_idx + sequence_length].values

        # Berechne Ähnlichkeit
        if len(compare_pos) == len(pattern_pos) and len(compare_curr) == len(pattern_curr):
            pos_diff = np.mean(np.abs(pattern_pos - compare_pos))
            curr_diff = np.mean(np.abs(pattern_curr - compare_curr))
            total_score = pos_diff + curr_diff

            if total_score < best_score:
                best_score = total_score
                best_match_idx = start_idx

    return best_match_idx, best_score

def map_data_to_reduced(data, data_reduced, tolerance=0.01):
    """
    Mappt data auf den entsprechenden Bereich in data_reduced
    """
    # Finde den Startindex von data in data_reduced
    start_idx, score = find_matching_sequence(data, data_reduced)

    print(f"Bester Match gefunden bei Index {start_idx} mit Score {score:.6f}")

    if score > tolerance:
        print("⚠️ Kein guter Match – nehme trotzdem best_idx zur Analyse.")
        score = tolerance

    if start_idx == -1 or score > tolerance:
        print("Warnung: Kein guter Match gefunden!")
        print(f"Bester Match Score: {score:.6f}, Threshold: {tolerance}")
        print(f"start_idx: {start_idx}")
        return data



    # Erstelle erweiterten DataFrame
    data_extended = data.copy()

    # Prüfe ob genügend Daten in data_reduced vorhanden sind
    end_idx = start_idx + len(data)
    if end_idx > len(data_reduced):
        print(
            f"Warnung: Nicht genügend Daten in data_reduced (benötigt {len(data)}, verfügbar {len(data_reduced) - start_idx})")
        end_idx = len(data_reduced)

    # Extrahiere die entsprechenden CONT_DEV Werte
    cont_dev_slice = data_reduced.iloc[start_idx:end_idx]

    # Füge die CONT_DEV Spalten zu data hinzu
    for col in ['CONT_DEV_X', 'CONT_DEV_Y', 'CONT_DEV_Z', 'CONT_DEV_SP']:
        if col in cont_dev_slice.columns:
            # Stelle sicher, dass die Längen übereinstimmen
            values_to_add = cont_dev_slice[col].values[:len(data)]
            data_extended[col] = values_to_add
        else:
            data_extended[col] = np.nan

    return data_extended

def alternative_correlation_mapping(data, data_reduced):
    """
    Alternative Methode: Findet Übereinstimmung durch Korrelationsanalyse
    """
    # Verwende Strom-Signale für Korrelation (meist kontinuierlicher)
    data_curr_signal = data['curr_x'].values
    reduced_curr_signal = data_reduced['CURRENT_X'].values

    # Sliding window Korrelation
    best_corr = -1
    best_idx = -1

    for start_idx in range(len(reduced_curr_signal) - len(data_curr_signal) + 1):
        window = reduced_curr_signal[start_idx:start_idx + len(data_curr_signal)]

        # Berechne Korrelation
        if len(window) == len(data_curr_signal):
            corr = np.corrcoef(data_curr_signal, window)[0, 1]
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_idx = start_idx

    print(f"Beste Korrelation: {best_corr:.4f} bei Index {best_idx}")

    if best_idx == -1:
        return data

    # Anwenden der Zuordnung
    data_extended = data.copy()
    end_idx = min(best_idx + len(data), len(data_reduced))

    cont_dev_slice = data_reduced.iloc[best_idx:end_idx]

    for col in ['CONT_DEV_X', 'CONT_DEV_Y', 'CONT_DEV_Z', 'CONT_DEV_SP']:
        if col in cont_dev_slice.columns:
            values_to_add = cont_dev_slice[col].values[:len(data)]
            data_extended[col] = values_to_add
        else:
            data_extended[col] = np.nan

    return data_extended

def plot_sequence_overlap(data, data_reduced, data_extended, filename):
    """
    Plottet die Überlappung zwischen data und data_reduced um die Zuordnung zu visualisieren
    """
    import matplotlib.pyplot as plt

    # Finde den Startindex erneut für den Plot
    start_idx, score = find_matching_sequence(data, data_reduced)

    if start_idx == -1:
        print("Kein Match gefunden - kann nicht plotten")
        return

    end_idx = min(start_idx + len(data), len(data_reduced))

    # Erstelle Subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Sequenz-Überlappung für {filename}\nMatch bei Index {start_idx}-{end_idx} (Score: {score:.6f})',
                 fontsize=14)

    # Plot 1: Position X
    axes[0, 0].plot(data.index, data['pos_x'], 'r-', label='data: pos_x', linewidth=2)
    axes[0, 0].plot(range(start_idx, end_idx), data_reduced['ENC_POS_X'].iloc[start_idx:end_idx],
                    'g--', label='data_reduced: ENC_POS_X (matched)', linewidth=2)
    axes[0, 0].plot(data_reduced.index, data_reduced['ENC_POS_X'], 'g-', alpha=0.3,
                    label='data_reduced: ENC_POS_X (full)')
    axes[0, 0].set_title('Position X Vergleich')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Position X')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Current X
    axes[0, 1].plot(data.index, data['curr_x'], 'r-', label='data: curr_x', linewidth=2)
    axes[0, 1].plot(range(start_idx, end_idx), data_reduced['CURRENT_X'].iloc[start_idx:end_idx],
                    'b--', label='data_reduced: CURRENT_X (matched)', linewidth=2)
    axes[0, 1].plot(data_reduced.index, data_reduced['CURRENT_X'], 'b-', alpha=0.3,
                    label='data_reduced: CURRENT_X (full)')
    axes[0, 1].set_title('Current X Vergleich')
    axes[0, 1].set_xlabel('Index')
    axes[0, 1].set_ylabel('Current X')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot 3: CONT_DEV_X (neu hinzugefügte Daten)
    if 'CONT_DEV_X' in data_extended.columns and data_extended['CONT_DEV_X'].notna().any():
        axes[1, 0].plot(data.index, data_extended['CONT_DEV_X'], 'purple', linewidth=2, label='Hinzugefügte CONT_DEV_X')
        axes[1, 0].plot(data_reduced.index, data_reduced['CONT_DEV_X'], 'purple', alpha=0.3,
                        label='data_reduced: CONT_DEV_X (full)')
        # Markiere den gematchten Bereich
        axes[1, 0].axvspan(start_idx, end_idx, alpha=0.2, color='yellow', label='Gematchter Bereich')
        axes[1, 0].set_title('CONT_DEV_X - Hinzugefügte Daten')
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel('CONT_DEV_X')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'Keine CONT_DEV_X Daten\nverfügbar oder gemappt',
                        transform=axes[1, 0].transAxes, ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('CONT_DEV_X - Nicht verfügbar')

    # Plot 4: Differenz zwischen gematchten Sequenzen
    if len(data) <= end_idx - start_idx:
        pos_diff = np.abs(data['pos_x'].values - data_reduced['ENC_POS_X'].iloc[start_idx:start_idx + len(data)].values)
        curr_diff = np.abs(
            data['curr_x'].values - data_reduced['CURRENT_X'].iloc[start_idx:start_idx + len(data)].values)

        axes[1, 1].plot(data.index, pos_diff, 'r-', label='|pos_x - ENC_POS_X|', linewidth=2)
        axes[1, 1].plot(data.index, curr_diff, 'b-', label='|curr_x - CURRENT_X|', linewidth=2)
        axes[1, 1].set_title('Absolute Differenzen (Matching-Qualität)')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Absolute Differenz')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')  # Log-Skala für bessere Sichtbarkeit kleiner Unterschiede
    else:
        axes[1, 1].text(0.5, 0.5, 'Längen-Mismatch\nzwischen data und\ndata_reduced slice',
                        transform=axes[1, 1].transAxes, ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Differenzen - Nicht berechenbar')

    plt.tight_layout()
    plt.show()

    # Zusätzliche Statistiken
    print(f"Mapping Statistiken:")
    print(f"  - data Länge: {len(data)}")
    print(f"  - Gematchter Bereich in data_reduced: {start_idx} bis {end_idx}")
    print(f"  - Bereich Länge: {end_idx - start_idx}")
    if 'CONT_DEV_X' in data_extended.columns:
        non_nan_count = data_extended['CONT_DEV_X'].notna().sum()
        print(
            f"  - Erfolgreich gemappte CONT_DEV Werte: {non_nan_count}/{len(data)} ({100 * non_nan_count / len(data):.1f}%)")

        if non_nan_count > 0:
            print(
                f"  - CONT_DEV_X Wertebereich: {data_extended['CONT_DEV_X'].min():.3f} bis {data_extended['CONT_DEV_X'].max():.3f}")
            print(f"  - CONT_DEV_X Mittelwert: {data_extended['CONT_DEV_X'].mean():.3f}")
            print(f"  - CONT_DEV_X Standardabweichung: {data_extended['CONT_DEV_X'].std():.3f}")

# Setup
path_additional_data = 'AdditionalDataFiltered'
path_data = 'DataFiltered'
path_target = 'DataMatched'
files = ['AL_2007_T4_Plate_Normal', 'S235JR_Plate_Normal']
files = os.listdir(path_additional_data)
# Hauptverarbeitung
for file in files:
    file = file.replace('.csv', '')
    print(f"\nVerarbeitung: {file}")

    for version in ['1', '2', '3']:
        print(f"\nVersion: {version}")
        data = pd.read_csv(f'{path_data}/{file}_{version}.csv')
        data_additional = pd.read_csv(f'{path_additional_data}/{file}.csv')

        # Preprocessing wie gewohnt
        window_size = 10
        data_smoothed = data_additional.rolling(window=window_size, min_periods=1).mean()
        data_reduced = data_smoothed.iloc[::window_size].reset_index(drop=True)

        print(f"data Länge: {len(data)}")
        print(f"data_reduced Länge: {len(data_reduced)}")

        # Methode 1: Sequenz-basiertes Matching
        print("\n=== Sequenz-basiertes Matching ===")
        data_extended_seq = map_data_to_reduced(data, data_reduced, tolerance=0.1)
        matches_seq = data_extended_seq['CONT_DEV_X'].notna().sum()
        print(f"Erfolgreich gemappte Zeilen: {matches_seq}/{len(data)}")

        # Methode 2: Korrelations-basiertes Matching
        print("\n=== Korrelations-basiertes Matching ===")
        data_extended_corr = alternative_correlation_mapping(data, data_reduced)
        matches_corr = data_extended_corr['CONT_DEV_X'].notna().sum()
        print(f"Erfolgreich gemappte Zeilen: {matches_corr}/{len(data)}")

        # Wähle die bessere Methode
        if matches_seq >= matches_corr:
            final_data = data_extended_seq
            print(f"\nVerwende Sequenz-Matching (besser)")
        else:
            final_data = data_extended_corr
            print(f"\nVerwende Korrelations-Matching (besser)")

        # Validierung: Prüfe ob die Zuordnung sinnvoll ist
        print("\n=== Validierung ===")
        if 'CONT_DEV_X' in final_data.columns:
            print(f"CONT_DEV_X Bereich: {final_data['CONT_DEV_X'].min():.3f} bis {final_data['CONT_DEV_X'].max():.3f}")
            print(f"Nicht-NaN Werte: {final_data['CONT_DEV_X'].notna().sum()}")

        # Optional: Speichere das Ergebnis
        final_data.to_csv(f'{path_target}/{file}_{version}.csv', index=False)

        # Plot der Überlappungsbereiche
        print("\n=== Visualisierung der Sequenz-Überlappung ===")
        plot_sequence_overlap(data, data_reduced, data_extended_seq, file)

    print("=" * 50)