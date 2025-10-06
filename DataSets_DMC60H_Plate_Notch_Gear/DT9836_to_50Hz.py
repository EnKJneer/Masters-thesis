import os
import numpy as np
import pandas as pd
from numpy.f2py.auxfuncs import throw_error
from scipy.interpolate import make_interp_spline
from scipy.signal import firwin, filtfilt

def time_to_float(time_str):
    """Convert time string to float."""
    return float(time_str.split()[0])

def apply_moving_average_filter(df, header, window_size):
    """Apply a moving average filter to the data, excluding the 'time' column and fill NaN values forward and backward."""
    df_filtered = df.copy()
    for column in header[1:]:
        if column != 'time':
            df_filtered[column] = df[column].rolling(window=window_size, center=True).mean()
            df_filtered[column] = df_filtered[column].ffill().bfill()
    return df_filtered

if __name__ == "__main__":
    header = ['time', 'DT_0', 'f_x', 'f_y', 'f_z', 'a_x', 'a_y', 'a_z']
    path_data = 'RawData'
    path_target = 'DataDT9836_50Hz'
    files = os.listdir(path_data)
    target_fs = 50
    cutoff_freq = 24

    for file in files:
        if not file.endswith('.csv'):
            continue
        print(f'Processing {file}')
        df = pd.read_csv(os.path.join(path_data, file))
        df.columns = header

        df['time'] = df['time'].apply(time_to_float)

        f_s_1 = (1 /(df['time'].iloc[1] - df['time'].iloc[0])).round(6)
        #print(f'Sampling frequency 1: {f_s_1}')

        duration = df['time'].max() -df['time'].min()
        print(f'Duration: {duration}')

        f_s_2 = (len(df['time']) / duration).round(6)
        #print(f'Sampling frequency 2: {f_s_2}')
        assert f_s_1 == f_s_2, 'Sampling frequency is unknown'

        #df = df.groupby('time').mean().reset_index()
        f_s = (f_s_1 + f_s_2)/2
        print(f'Sampling frequency: {f_s}')
        df['time'] = df.index * 1/f_s

        downsampling_factor = int(round(f_s / target_fs))
        print(f'Down sampling factor: {downsampling_factor}')
        print(f'New sampling frequency: {f_s / downsampling_factor}')
        if downsampling_factor > 1:
            df_filtered = apply_moving_average_filter(df, header, int(downsampling_factor)) #/2

            df_resampled = df_filtered.iloc[::downsampling_factor].reset_index(drop=True)
            df_resampled['time'] = df_resampled.index * 1 / target_fs
        else:
            df_resampled = df

        file = file.replace('AL_2007_T4', 'AL2007T4')
        file_path = os.path.join(path_target, file)
        print(f'Saving {file_path}')
        df_resampled.to_csv(file_path, index=False)
