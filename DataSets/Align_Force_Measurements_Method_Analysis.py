import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
def align_and_merge_data(data: pd.DataFrame, data_additional: pd.DataFrame, max_shift: int = 200):
    def calculate_mse_for_offset(offset: int) -> float:
        shifted = data_additional.shift(offset)
        valid = data[['pos_x', 'pos_y', 'pos_z', 'curr_x', 'curr_y', 'curr_z']].notna().all(axis=1) & \
                shifted[['ENC_POS_X', 'ENC_POS_Y', 'ENC_POS_Z', 'CURRENT_X', 'CURRENT_Y', 'CURRENT_Z']].notna().all(axis=1)

        if not valid.any():
            return np.inf

        mse = np.mean([
            np.mean((data.loc[valid, 'pos_x'] - shifted.loc[valid, 'ENC_POS_X']) ** 2),
            np.mean((data.loc[valid, 'pos_y'] - shifted.loc[valid, 'ENC_POS_Y']) ** 2),
            np.mean((data.loc[valid, 'pos_z'] - shifted.loc[valid, 'ENC_POS_Z']) ** 2),
            np.mean((data.loc[valid, 'curr_x'] - shifted.loc[valid, 'CURRENT_X']) ** 2),
            np.mean((data.loc[valid, 'curr_y'] - shifted.loc[valid, 'CURRENT_Y']) ** 2),
            np.mean((data.loc[valid, 'curr_z'] - shifted.loc[valid, 'CURRENT_Z']) ** 2),
        ])
        return mse

    offsets = list(range(-max_shift, max_shift + 1, 20))
    scores = [calculate_mse_for_offset(o) for o in offsets]
    best_offset = offsets[np.argmin(scores)]

    offsets_hr = list(range(best_offset - 10, best_offset + 11, 1))
    scores_hr = [calculate_mse_for_offset(o) for o in offsets_hr]
    best_offset = offsets_hr[np.argmin(scores_hr)]

    offsets_all = offsets + offsets_hr
    mse_scores_all = scores + scores_hr

    aligned_additional = data_additional.shift(best_offset).reset_index(drop=True)
    aligned_additional = aligned_additional.reindex(index=data.index)

    # Zeitlicher MSE-Verlauf pro Zeitschritt
    def compute_stepwise_mse():
        mse_series = []
        for i in range(len(data)):
            try:
                values_data = np.array([
                    data.loc[i, 'pos_x'],
                    data.loc[i, 'pos_y'],
                    data.loc[i, 'pos_z'],
                    data.loc[i, 'curr_x'],
                    data.loc[i, 'curr_y'],
                    data.loc[i, 'curr_z'],
                ])
                values_add = np.array([
                    aligned_additional.loc[i, 'ENC_POS_X'],
                    aligned_additional.loc[i, 'ENC_POS_Y'],
                    aligned_additional.loc[i, 'ENC_POS_Z'],
                    aligned_additional.loc[i, 'CURRENT_X'],
                    aligned_additional.loc[i, 'CURRENT_Y'],
                    aligned_additional.loc[i, 'CURRENT_Z'],
                ])
                if np.any(pd.isna(values_data)) or np.any(pd.isna(values_add)):
                    mse_series.append(np.nan)
                else:
                    mse_series.append(np.mean((values_data - values_add) ** 2))
            except:
                mse_series.append(np.nan)
        return pd.Series(mse_series, index=data.index)

    mapping = {
        'CONT_DEV_X': 'controler_dev_x',
        'CONT_DEV_Y': 'controler_dev_y',
        'CONT_DEV_Z': 'controler_dev_z',
        'CONT_DEV_SP': 'controler_dev_sp',
    }

    mse_timeline = compute_stepwise_mse()

    data_merged = data.copy()
    for old_col, new_col in mapping.items():
        if old_col in aligned_additional.columns:
            data_merged[new_col] = aligned_additional[old_col]
        else:
            print(f"Achtung: Spalte '{old_col}' nicht in data_additional gefunden.")

    return data_merged, aligned_additional, best_offset, mse_timeline



# Hauptverarbeitung
path_additional_data = 'OldData_Additional'
path_data = 'OldData'
path_target = 'OldData_Controller'
os.makedirs(path_target, exist_ok=True)

files = os.listdir(path_additional_data)

for file in files:
    if not file.endswith('.csv'):
        continue
    file = file.replace('.csv', '')
    print(f"\nVerarbeitung: {file}")

    data = pd.read_csv(f'{path_data}/{file}.csv')
    data_additional = pd.read_csv(f'{path_additional_data}/{file}.csv')

    window_size = 10
    #data_smoothed = data_additional.rolling(window=window_size, min_periods=1).mean()
    data_reduced = data_additional.iloc[::window_size].reset_index(drop=True)

    print(f"data Länge: {len(data)}")
    print(f"data_reduced Länge: {len(data_reduced)}")

    data_merged, aligned_additional, best_offset, mse_timeline = align_and_merge_data(data, data_reduced)

    # Speichern
    target_path = os.path.join(path_target, f"{file}.csv")
    data_merged.to_csv(target_path, index=False)
    print(f"Gespeichert: {target_path}")

    # Plot zur Prüfung
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(data['pos_x'], label='pos_x (data)')
    axs[0].plot(aligned_additional['ENC_POS_X'], label='ENC_POS_X (aligned)', alpha=0.7)
    axs[0].legend()
    axs[0].set_title('Position X')

    axs[1].plot(data['curr_x'], label='curr_x')
    axs[1].plot(aligned_additional['CURRENT_X'], label='CURRENT_X (aligned)', alpha=0.7)
    axs[1].legend()
    axs[1].set_title('Strom X')

    axs[2].plot(data_merged['controler_dev_x'], label='controler_dev_x')
    axs[2].set_title('Übernommene controler_dev_x')

    plt.suptitle(f'Alignment-Check: {file}')
    plt.tight_layout()
    plot_path_target = os.path.join(path_target, f"Plots")
    plot_path = os.path.join(plot_path_target, f"{file}_alignment_check.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot gespeichert: {plot_path}")

    # Zeitlicher Verlauf des MSE (bei best_offset)
    plt.figure(figsize=(12, 4))
    plt.plot(mse_timeline, label='MSE pro Zeitschritt')
    plt.title(f'MSE-Zeitverlauf bei optimalem Offset ({best_offset}) – {file}')
    plt.xlabel('Zeitindex')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    mse_time_path = os.path.join(plot_path_target, f"{file}_mse_time_series.png")
    plt.savefig(mse_time_path)
    plt.close()
    print(f"Zeitlicher MSE-Plot gespeichert: {mse_time_path}")


