import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def detect_peaks_and_split(df, time_col='time', signal_col='curr_z',
                           plot=False, filename='', path_target = 'DataSets_CMX_Plate_Notch_Gear\Data\Plots'):
    """
    Detects 2-4 peaks in time series data and splits DataFrame into regions.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing time series data
    time_col : str
        Column name for time data
    signal_col : str
        Column name for signal data (curr_z)
    plot : bool
        Whether to create a plot showing peaks and regions
    filename : str
        Filename for plot title and saving
    path_target : str
        Path where plot will be saved.

    Returns:
    --------
    dict : Dictionary with keys 'peaks', 'regions', 'df_parts'
           - peaks: indices of detected peaks
           - regions: list of region boundaries
           - df_parts: list of DataFrame parts
    """

    # Extract time and signal data
    time_data = df[time_col].values
    signal_data = df[signal_col].values

    # Peak detection parameters
    # Based on the plot, peaks are very prominent and narrow
    height_threshold = np.std(signal_data) * 2  # Peaks should be at least 2 std above noise
    min_distance = len(signal_data) // 12  # Minimum distance between peaks (1/12 of total length)

    # Find only positive peaks - they come in pairs with negative peaks
    positive_peaks, _ = find_peaks(signal_data,
                                   height=height_threshold,
                                   distance=min_distance)

    # Use only positive peaks
    all_peaks = positive_peaks

    # Filter peaks by magnitude (keep only the most significant ones)
    if len(all_peaks) > 0:
        peak_magnitudes = np.abs(signal_data[all_peaks])
        # Keep peaks with magnitude > 12.5% of max magnitude
        magnitude_threshold = np.max(peak_magnitudes) * 0.125
        significant_peaks = all_peaks[peak_magnitudes > magnitude_threshold]
    else:
        significant_peaks = all_peaks

    # Ensure we have 2-4 peaks as expected
    if len(significant_peaks) < 2:
        print(f"Warning: Only {len(significant_peaks)} peaks found in {filename}")
    elif len(significant_peaks) > 4:
        # Keep the 4 most significant peaks
        peak_magnitudes = np.abs(signal_data[significant_peaks])
        top_indices = np.argsort(peak_magnitudes)[-4:]
        significant_peaks = np.sort(significant_peaks[top_indices])
        print(f"Warning: {len(all_peaks)} peaks found, keeping 4 most significant")

    peaks = significant_peaks

    # Define regions based on peaks
    # Region naming: 0 (optional start), 1,2,3 (main regions), 4 (optional end)
    regions = []
    region_names = []

    if len(peaks) == 0:
        # No peaks found, treat as single main region
        regions.append((0, len(signal_data) - 1))
        region_names.append(1)
    elif len(peaks) == 2:
        # 2 peaks = 3 main regions (1, 2, 3)
        regions.append((0, peaks[0]))
        regions.append((peaks[0], peaks[1]))
        regions.append((peaks[1], len(signal_data) - 1))
        region_names = [1, 2, 3]
    elif len(peaks) == 3:
        # 3 peaks = 4 regions: could be (0,1,2,3) or (1,2,3,4)
        # Check if first region is significantly smaller (likely region 0)
        first_region_size = peaks[0]
        total_size = len(signal_data)

        if first_region_size < total_size * 0.15:  # First region < 15% of total
            # Pattern: 0, 1, 2, 3
            regions.append((0, peaks[0]))
            regions.append((peaks[0], peaks[1]))
            regions.append((peaks[1], peaks[2]))
            regions.append((peaks[2], len(signal_data) - 1))
            region_names = [0, 1, 2, 3]
        else:
            # Pattern: 1, 2, 3, 4
            regions.append((0, peaks[0]))
            regions.append((peaks[0], peaks[1]))
            regions.append((peaks[1], peaks[2]))
            regions.append((peaks[2], len(signal_data) - 1))
            region_names = [1, 2, 3, 4]
    elif len(peaks) == 4:
        # 4 peaks = 5 regions (0, 1, 2, 3, 4)
        regions.append((0, peaks[0]))
        regions.append((peaks[0], peaks[1]))
        regions.append((peaks[1], peaks[2]))
        regions.append((peaks[2], peaks[3]))
        regions.append((peaks[3], len(signal_data) - 1))
        region_names = [0, 1, 2, 3, 4]
    else:
        # More than 4 peaks - fallback to sequential numbering
        regions.append((0, peaks[0]))
        for i in range(len(peaks) - 1):
            regions.append((peaks[i], peaks[i + 1]))
        regions.append((peaks[-1], len(signal_data) - 1))
        region_names = list(range(len(regions)))

    # Split DataFrame into parts based on regions
    df_parts = {}  # Use dict with region names as keys
    for i, ((start_idx, end_idx), region_name) in enumerate(zip(regions, region_names)):
        # Add some overlap to avoid losing data at boundaries
        actual_start = max(0, start_idx - 10) if i > 0 else 0
        actual_end = min(len(df) - 1, end_idx + 10) if i < len(regions) - 1 else len(df) - 1

        df_part = df.iloc[actual_start:actual_end + 1].copy()
        df_parts[region_name] = df_part

    # Create plot if requested
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time_data, signal_data, 'g-', label=signal_col, linewidth=1)

        # Mark peaks
        if len(peaks) > 0:
            plt.plot(time_data[peaks], signal_data[peaks], 'ro',
                     markersize=8, label=f'Detected Peaks ({len(peaks)})')

            # Number the peaks
            for i, peak_idx in enumerate(peaks, 1):
                plt.annotate(f'{i}', (time_data[peak_idx], signal_data[peak_idx]),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=12, fontweight='bold', color='red')

        # Mark regions with vertical lines and colors
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink']
        region_color_map = {0: 'lightcoral', 1: 'lightblue', 2: 'lightgreen', 3: 'lightyellow', 4: 'lightpink'}

        for i, ((start_idx, end_idx), region_name) in enumerate(zip(regions, region_names)):
            color = region_color_map.get(region_name, colors[i % len(colors)])
            plt.axvspan(time_data[start_idx], time_data[end_idx],
                        alpha=0.3, color=color,
                        label=f'Region {region_name}')

        plt.xlabel('Time in s')
        plt.ylabel('curr_z')
        plt.title(f'{filename} - Peak Detection and Region Splitting')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        if filename:
            os.makedirs(path_target, exist_ok=True)
            path = os.path.join(path_target, f'{filename}_peaks_regions.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')

        if show_plots:
            plt.show()

    return {
        'peaks': peaks,
        'regions': regions,
        'region_names': region_names,
        'df_parts': df_parts,
        'peak_times': time_data[peaks] if len(peaks) > 0 else [],
        'peak_values': signal_data[peaks] if len(peaks) > 0 else []
    }


def get_filname(file: str) -> str:
        filename = file.split('.')[0]
        filename = 'DMC60H_'+filename
        filename = filename.replace('AL_2007_T4', 'AL2007T4')
        return filename

if __name__ == "__main__":
    path_data = 'DataSimulated'
    path_target = 'Data_1'

    # Create target directory if it doesn't exist
    os.makedirs(path_target, exist_ok=True)

    files = os.listdir(path_data)

    show_plots = True

    for file in files:
        if not file.endswith('.csv'):
            continue

        print(f'Processing {file}')
        df = pd.read_csv(os.path.join(path_data, file))
        filename = get_filname(file)

        # Detect peaks and split data
        result = detect_peaks_and_split(df, plot=show_plots, filename=filename)

        df_parts = result['df_parts']  # Now a dictionary with region names as keys
        peaks = result['peaks']
        region_names = result['region_names']

        print(f'  Found {len(peaks)} peaks')
        print(f'  Split into regions: {region_names}')

        # Save each part as a separate CSV file using correct region names
        for region_name, part in df_parts.items():
            if not (filename.startswith('DMC_AL') and filename.endswith('Depth') and region_name == 3):
                output_filename = f'{filename}_{region_name}.csv'
            else:
                # Bei Alu depth enthält das letzte 3 Anomalien wie fressen und gebrochene Fräser
                output_filename = f'{filename}_Ano.csv'

            if not (region_name == 0 or region_name == 4):
                part = part.reset_index(drop=True)
                # Filter nur die Prozessdaten raus. -> materialremoved_sim > 0 +/- 5% der Datenlänge
                indices = part[part['materialremoved_sim'] > 100].index #Problem: MRR Berechnung ist zu ungenau.

                n = indices.max() - indices.min()
                p = 0.01
                start_index = max(0, indices.min() - int(p * n))
                end_index = min(len(part), indices.max() + int(p * n))
                part = part.iloc[start_index:end_index] #.reset_index(drop=True)


                plt.plot(part['materialremoved_sim'], label='Material Removed Sim')
                plt.plot(part['curr_x'] * 100, label='Current X')
                plt.axvline(x=start_index, color='r', linestyle='--', label='Start Index')
                plt.axvline(x=end_index, color='g', linestyle='--', label='End Index')
                plt.legend()

                if show_plots:
                    plt.show()

                part = part.reset_index(drop=True)
            else:
                output_filename = f'{filename}_Aircut_{region_name}.csv'
            part.to_csv(os.path.join(path_target, output_filename), index=False)
            print(f'  Saved {output_filename} with {len(part)} rows')

        print()