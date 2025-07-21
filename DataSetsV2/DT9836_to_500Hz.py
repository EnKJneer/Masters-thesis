import os
import numpy as np
import pandas as pd
from numpy.f2py.auxfuncs import throw_error
from scipy.interpolate import make_interp_spline
from scipy.signal import firwin, filtfilt

def time_to_float(time_str):
    """Convert time string to float."""
    return float(time_str.split()[0])

def apply_fir_filter(df, header, cutoff_freq, fs):
    """Apply FIR filter to the data."""
    numtaps = 101
    fir_coeff = firwin(numtaps, cutoff_freq, fs=fs)
    df_filtered = df.copy()
    for column in header[1:]:
        df_filtered[column] = filtfilt(fir_coeff, 1.0, df[column])
    return df_filtered

def apply_moving_average_filter(df, header, window_size):
    """Apply a moving average filter to the data, excluding the 'Time' column and fill NaN values forward and backward."""
    df_filtered = df.copy()
    for column in header[1:]:
        if column != 'Time':
            df_filtered[column] = df[column].rolling(window=window_size, center=True).mean()
            df_filtered[column] = df_filtered[column].ffill().bfill()
    return df_filtered

def resample_data(df, header, target_fs):
    """Resample data to target sampling frequency using spline interpolation."""
    time_old = df['Time'].values
    time_new = np.arange(time_old[0], time_old[-1], 1/target_fs)
    df_resampled = pd.DataFrame({'Time': time_new})

    for column in header[1:]:
        spline = make_interp_spline(time_old, df[column].values, k=3)
        df_resampled[column] = spline(time_new)

    return df_resampled

def process_segment(segment, header, target_fs, cutoff_freq=249):
    """Process a segment of data: filter and resample."""
    segment_time_diff = segment['Time'].diff().dropna().round(6)
    segment_fs = (1 / segment_time_diff.mean()).round(6)
    print(f'Segment sampling frequency: {segment_fs} Hz')

    if segment_fs >= target_fs:
        if cutoff_freq > target_fs / 2:
            raise ValueError(f"Invalid cutoff frequency: {cutoff_freq}. It must be less than {target_fs / 2}.")
        segment_filtered = apply_fir_filter(segment, header, cutoff_freq, segment_fs)
    else:
        print("Segment sampling frequency is too high for filtering.")
        segment_filtered = segment
    window_size = int(segment_fs / target_fs) *10
    if window_size > 1:
        segment_filtered = apply_moving_average_filter(segment_filtered, header, window_size)
    segment_resampled = resample_data(segment_filtered, header, target_fs)

    return segment_resampled


if __name__ == "__main__":
    header = ['Time', 'DT_0', 'f_x', 'f_y', 'f_z', 'a_x', 'a_y', 'a_z']
    path_data = 'RawData'
    path_target = 'RawDataDT9836'
    files = os.listdir(path_data)
    target_fs = 500
    cutoff_freq = 249

    for file in files:
        if not file.endswith('.csv'):
            continue
        print(f'Processing {file}')
        df = pd.read_csv(os.path.join(path_data, file))
        df.columns = header

        df['Time'] = df['Time'].apply(time_to_float)

        f_s_1 = (1 /(df['Time'].iloc[1] - df['Time'].iloc[0])).round(6)
        #print(f'Sampling frequency 1: {f_s_1}')

        duration = df['Time'].max() -df['Time'].min()
        print(f'Duration: {duration}')

        f_s_2 = (len(df['Time']) / duration).round(6)
        #print(f'Sampling frequency 2: {f_s_2}')
        if f_s_2 != f_s_1:
            print('Sampling frequency is unknown')
            throw_error('Sampling frequency unknown')
        #df = df.groupby('Time').mean().reset_index()
        f_s = (f_s_1 + f_s_2)/2
        print(f'Sampling frequency: {f_s}')
        df['Time'] = df.index * 1/f_s

        downsampling_factor = int(round(f_s / target_fs))
        print(f'Down sampling factor: {downsampling_factor}')
        print(f'New sampling frequency: {f_s / downsampling_factor}')
        if downsampling_factor > 1:

            df_filtered = apply_fir_filter(df, header, cutoff_freq, f_s)
            df_filtered = apply_moving_average_filter(df_filtered, header, int(downsampling_factor/2))

            df_resampled = df_filtered.iloc[::downsampling_factor].reset_index(drop=True)
            df_resampled['Time'] = df_resampled.index * 1 / target_fs
        else:
            df_resampled = df
        #name = file.replace('.csv', '_DT9836.csv')
        file_path = os.path.join(path_target, file)
        print(f'Saving {file_path}')
        df_resampled.to_csv(file_path, index=False)
