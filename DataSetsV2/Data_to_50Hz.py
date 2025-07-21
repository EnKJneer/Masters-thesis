import os
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.signal import firwin, filtfilt

def apply_fir_filter(df, cutoff_freq, fs):
    """Apply FIR filter to the data."""
    numtaps = 101
    fir_coeff = firwin(numtaps, cutoff_freq, fs=fs)
    df_filtered = df.copy()
    df_filtered = df_filtered.apply(lambda x: filtfilt(fir_coeff, 1.0, x))
    return df_filtered

def apply_moving_average_filter(df, window_size):
    """Apply a moving average filter to the data and fill NaN values forward and backward."""
    df_filtered = df.copy()
    df_filtered = df_filtered.rolling(window=window_size, center=True).mean()
    df_filtered = df_filtered.ffill().bfill()
    return df_filtered

def resample_data(df, target_fs, current_fs):
    """Resample data to target sampling frequency using spline interpolation."""
    time_old = df.index * 1/current_fs
    time_new = np.arange(time_old[0], time_old[-1], 1/target_fs)
    df_resampled = pd.DataFrame()
    for column in df.columns:
        spline = make_interp_spline(time_old, df[column].values, k=3)
        df_resampled[column] = spline(time_new)
    return df_resampled

if __name__ == "__main__":
    path_data = 'MergedData'
    path_target = 'Data'
    files = os.listdir(path_data)
    fs_target = 50
    cutoff_freq = 24
    fs_current = 500

    for file in files:
        if not file.endswith('.csv'):
            continue

        print(f'Processing {file}')
        df = pd.read_csv(os.path.join(path_data, file))

        downsampling_factor = int(round(fs_current / fs_target))
        print(f'Down sampling factor: {downsampling_factor}')
        print(f'New sampling frequency: {fs_current / downsampling_factor}')

        if downsampling_factor > 1:
            df_filtered = apply_fir_filter(df, cutoff_freq, fs_target)
            df_filtered = apply_moving_average_filter(df_filtered, int(downsampling_factor/2))
            df_resampled = df_filtered.iloc[::downsampling_factor].reset_index(drop=True)
        else:
            df_resampled = df

        filename = file.split('.')[0]

        # Split the DataFrame into three parts
        df_parts = np.array_split(df_resampled, 3)

        # Save each part as a separate CSV file
        for i, part in enumerate(df_parts, start=1):
            part.to_csv(os.path.join(path_target, f'{filename}_{i}.csv'), index=False)
