import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def calculate_mse(data1, data2):
    """Berechnet den durchschnittlichen MSE zwischen den linearen Regressionen mehrerer Signale."""
    mse_sum = 0
    count = 0

    for col in data1.columns:
        signal1 = data1[col].values.reshape(-1, 1)
        signal2 = data2[col].values.reshape(-1, 1)

        valid_idx1 = ~np.isnan(signal1)
        valid_idx2 = ~np.isnan(signal2)

        if np.sum(valid_idx1) < 2 or np.sum(valid_idx2) < 2:
            continue

        s1, s2 = signal1[valid_idx1], signal2[valid_idx2]

        x1, x2 = np.arange(len(s1)).reshape(-1, 1), np.arange(len(s2)).reshape(-1, 1)

        model1, model2 = LinearRegression().fit(x1, s1), LinearRegression().fit(x2, s2)

        pred1, pred2 = model1.predict(x1), model2.predict(x2)

        min_len = min(len(pred1), len(pred2))
        mse_sum += mean_squared_error(pred1[:min_len], pred2[:min_len])
        count += 1

    return mse_sum / count if count else np.inf

def find_matching_sequence(data, data_reduced, sequence_length=10):
    """Findet die Position von `data` in `data_reduced` basierend auf dem MSE der linearen Regression."""
    pattern_pos = data[['curr_x', 'curr_y', 'curr_z', 'curr_sp']].iloc[:sequence_length]
    best_match_idx, best_score = -1, float('inf')

    for start_idx in range(len(data_reduced) - len(data) + 1):
        compare_pos = data_reduced[['CURRENT_X', 'CURRENT_Y', 'CURRENT_Z', 'CURRENT_SP']].iloc[start_idx:start_idx + len(data)]

        if len(compare_pos) == len(pattern_pos):
            mse = calculate_mse(pattern_pos, compare_pos)
            if mse < best_score:
                best_score, best_match_idx = mse, start_idx

    return best_match_idx, best_score

def map_data_to_reduced(data, data_reduced, tolerance=0.01):
    """Mappt `data` auf den entsprechenden Bereich in `data_reduced`."""
    start_idx, score = find_matching_sequence(data, data_reduced)
    print(f"Bester Match gefunden bei Index {start_idx} mit Score {score:.6f}")

    if start_idx == -1 or score > tolerance:
        print("Warnung: Kein guter Match gefunden!")
        return data

    data_extended = data.copy()
    end_idx = min(start_idx + len(data), len(data_reduced))
    cont_dev_slice = data_reduced.iloc[start_idx:end_idx]

    for col in ['CONT_DEV_X', 'CONT_DEV_Y', 'CONT_DEV_Z', 'CONT_DEV_SP']:
        data_extended[col] = cont_dev_slice[col].values[:len(data)] if col in cont_dev_slice.columns else np.nan

    return data_extended

def plot_sequence_overlap(data, data_reduced, data_extended, filename):
    """Plottet die Überlappung zwischen `data` und `data_reduced`."""
    start_idx, score = find_matching_sequence(data, data_reduced)
    if start_idx == -1:
        print("Kein Match gefunden - kann nicht plotten")
        return

    end_idx = min(start_idx + len(data), len(data_reduced))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Sequenz-Überlappung für {filename}\nMatch bei Index {start_idx}-{end_idx} (Score: {score:.6f})', fontsize=14)

    axes[0, 0].plot(data.index, data['pos_x'], 'r-', label='data: pos_x', linewidth=2)
    axes[0, 0].plot(range(start_idx, end_idx), data_reduced['ENC_POS_X'].iloc[start_idx:end_idx], 'g--', label='data_reduced: ENC_POS_X (matched)', linewidth=2)
    axes[0, 0].plot(data_reduced.index, data_reduced['ENC_POS_X'], 'g-', alpha=0.3, label='data_reduced: ENC_POS_X (full)')
    axes[0, 0].set_title('Position X Vergleich')
    axes[0, 0].set_xlabel('Index')
    axes[0, 0].set_ylabel('Position X')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(data.index, data['curr_x'], 'r-', label='data: curr_x', linewidth=2)
    axes[0, 1].plot(range(start_idx, end_idx), data_reduced['CURRENT_X'].iloc[start_idx:end_idx], 'b--', label='data_reduced: CURRENT_X (matched)', linewidth=2)
    axes[0, 1].plot(data_reduced.index, data_reduced['CURRENT_X'], 'b-', alpha=0.3, label='data_reduced: CURRENT_X (full)')
    axes[0, 1].set_title('Current X Vergleich')
    axes[0, 1].set_xlabel('Index')
    axes[0, 1].set_ylabel('Current X')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    if 'CONT_DEV_X' in data_extended.columns and data_extended['CONT_DEV_X'].notna().any():
        axes[1, 0].plot(data.index, data_extended['CONT_DEV_X'], 'purple', linewidth=2, label='Hinzugefügte CONT_DEV_X')
        axes[1, 0].plot(data_reduced.index, data_reduced['CONT_DEV_X'], 'purple', alpha=0.3, label='data_reduced: CONT_DEV_X (full)')
        axes[1, 0].axvspan(start_idx, end_idx, alpha=0.2, color='yellow', label='Gematchter Bereich')
        axes[1, 0].set_title('CONT_DEV_X - Hinzugefügte Daten')
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel('CONT_DEV_X')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'Keine CONT_DEV_X Daten verfügbar oder gemappt', ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('CONT_DEV_X - Nicht verfügbar')

    if len(data) <= end_idx - start_idx:
        pos_diff = np.abs(data['pos_x'].values - data_reduced['ENC_POS_X'].iloc[start_idx:start_idx + len(data)].values)
        curr_diff = np.abs(data['curr_x'].values - data_reduced['CURRENT_X'].iloc[start_idx:start_idx + len(data)].values)
        axes[1, 1].plot(data.index, pos_diff, 'r-', label='|pos_x - ENC_POS_X|', linewidth=2)
        axes[1, 1].plot(data.index, curr_diff, 'b-', label='|curr_x - CURRENT_X|', linewidth=2)
        axes[1, 1].set_title('Absolute Differenzen (Matching-Qualität)')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Absolute Differenz')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
    else:
        axes[1, 1].text(0.5, 0.5, 'Längen-Mismatch zwischen data und data_reduced slice', ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Differenzen - Nicht berechenbar')

    plt.tight_layout()
    plt.show()

# Hauptverarbeitung
path_additional_data = 'OldData_Additional'
path_data = 'OldData_Aligned'
path_target = 'OldData_Controller'
files = os.listdir(path_additional_data)

for file in files:
    file = file.replace('.csv', '')
    print(f"\nVerarbeitung: {file}")
    data = pd.read_csv(f'{path_data}/{file}_1.csv')
    data_additional = pd.read_csv(f'{path_additional_data}/{file}.csv')

    window_size = 10
    data_smoothed = data_additional.rolling(window=window_size, min_periods=1).mean()
    data_reduced = data_smoothed.iloc[::window_size].reset_index(drop=True)

    print(f"data Länge: {len(data)}")
    print(f"data_reduced Länge: {len(data_reduced)}")

    data_extended_seq = map_data_to_reduced(data, data_reduced, tolerance=0.1)
    matches_seq = data_extended_seq['CONT_DEV_X'].notna().sum()
    print(f"Erfolgreich gemappte Zeilen: {matches_seq}/{len(data)}")

    plot_sequence_overlap(data, data_reduced, data_extended_seq, file)
