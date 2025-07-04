import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

# Define the linear function for fitting
def linear_func(x, a, b):
    return a * x + b

# Define colors and labels for each file
file_colors = {
    'AL_2007_T4_Plate_Depth_1.csv': ('lightgreen', 'AL Plate Depth 1'),
    'AL_2007_T4_Gear_Depth_1.csv': ('mediumblue', 'AL Gear Depth 1'),
    'AL_2007_T4_Plate_Depth_2.csv': ('darkgreen', 'AL Plate Depth 2'),
    'AL_2007_T4_Gear_Depth_2.csv': ('navy', 'AL Gear Depth 2'),
    'AL_2007_T4_Plate_Depth_3.csv': ('olivedrab', 'AL Plate Depth 3'),
    'AL_2007_T4_Gear_Depth_3.csv': ('darkblue', 'AL Gear Depth 3'),
    'S235JR_Plate_Depth_1.csv': ('sandybrown', 'S Plate Depth 1'),
    'S235JR_Gear_Depth_1.csv': ('lightgray', 'S Gear Depth 1'),
    'S235JR_Plate_Depth_2.csv': ('brown', 'S Plate Depth 2'),
    'S235JR_Gear_Depth_2.csv': ('gray', 'S Gear Depth 2'),
    'S235JR_Plate_Depth_3.csv': ('saddlebrown', 'S Plate Depth 3'),
    'S235JR_Gear_Depth_3.csv': ('darkgray', 'S Gear Depth 3'),
    'AL_2007_T4_Plate_Normal_1.csv': ('blue', 'AL Plate 1'),
    'AL_2007_T4_Gear_Normal_1.csv': ('green', 'AL Gear 1'),
    'AL_2007_T4_Plate_Normal_2.csv': ('blue', 'AL Plate 2'),
    'AL_2007_T4_Gear_Normal_2.csv': ('green', 'AL Gear 2'),
    'AL_2007_T4_Plate_Normal_3.csv': ('blue', 'AL Plate 3'),
    'AL_2007_T4_Gear_Normal_3.csv': ('green', 'AL Gear 3'),
    'AL_2007_T4_Plate_SF_1.csv': ('cyan', 'AL Plate SF 1'),
    'AL_2007_T4_Gear_SF_1.csv': ('teal', 'AL Gear SF 1'),
    'AL_2007_T4_Plate_SF_2.csv': ('cyan', 'AL Plate SF 2'),
    'AL_2007_T4_Gear_SF_2.csv': ('teal', 'AL Gear SF 2'),
    'AL_2007_T4_Plate_SF_3.csv': ('cyan', 'AL Plate SF 3'),
    'AL_2007_T4_Gear_SF_3.csv': ('teal', 'AL Gear SF 3'),
    'S235JR_Gear_Normal_1.csv': ('orange', 'S Gear 1'),
    'S235JR_Plate_Normal_1.csv': ('red', 'S Plate 1'),
    'S235JR_Plate_Normal_2.csv': ('orange', 'S Plate 2'),
    'S235JR_Gear_Normal_2.csv': ('red', 'S Gear 2'),
    'S235JR_Plate_Normal_3.csv': ('orange', 'S Plate 3'),
    'S235JR_Gear_Normal_3.csv': ('red', 'S Gear 3'),
    'S235JR_Plate_SF_1.csv': ('pink', 'S Plate SF 1'),
    'S235JR_Gear_SF_1.csv': ('purple', 'S Gear SF 1'),
    'S235JR_Plate_SF_2.csv': ('pink', 'S Plate SF 2'),
    'S235JR_Gear_SF_2.csv': ('purple', 'S Gear SF 2'),
    'S235JR_Plate_SF_3.csv': ('pink', 'S Plate SF 3'),
    'S235JR_Gear_SF_3.csv': ('purple', 'S Gear SF 3'),
}

path_data = 'DataFiltered'
files = [
    #'AL_2007_T4_Plate_Normal_1.csv', 'AL_2007_T4_Gear_Normal_1.csv',
    #'AL_2007_T4_Plate_SF_1.csv', 'AL_2007_T4_Gear_SF_1.csv',
    #'AL_2007_T4_Plate_Depth_1.csv', 'AL_2007_T4_Gear_Depth_1.csv',
    'AL_2007_T4_Plate_Depth_2.csv', 'AL_2007_T4_Gear_Depth_2.csv',
    #'AL_2007_T4_Plate_Depth_3.csv', 'AL_2007_T4_Gear_Depth_3.csv',
    #'S235JR_Gear_Normal_1.csv', 'S235JR_Plate_Normal_1.csv',
    #'S235JR_Plate_SF_1.csv', 'S235JR_Gear_SF_1.csv',
    #'S235JR_Plate_Depth_1.csv', 'S235JR_Gear_Depth_1.csv',
    'S235JR_Plate_Depth_2.csv', 'S235JR_Gear_Depth_2.csv',
    #'S235JR_Plate_Depth_3.csv', 'S235JR_Gear_Depth_3.csv',
    'AL_2007_T4_Plate_Normal_2.csv', 'AL_2007_T4_Gear_Normal_2.csv',
    'AL_2007_T4_Plate_SF_2.csv', 'AL_2007_T4_Gear_SF_2.csv',
    #'AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv',
    #'AL_2007_T4_Plate_SF_3.csv', 'AL_2007_T4_Gear_SF_3.csv',
    'S235JR_Gear_Normal_2.csv', 'S235JR_Plate_Normal_2.csv',
    'S235JR_Plate_SF_2.csv', 'S235JR_Gear_SF_2.csv',
    #'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv',
    #'S235JR_Plate_SF_3.csv', 'S235JR_Gear_SF_3.csv'
]

n = 25

axes = ['x']


for axis in axes:
    # Create a figure with three subplots for this axis
    fig, axs = plt.subplots(1, 3, figsize=(21, 6), dpi=300)

    # Data storage for the table
    results = []

    # Dictionary to store the parameters
    parameters = {}

    for file in files:
        data = pd.read_csv(f'{path_data}/{file}')
        epsilon = 1e-2
        v_axis = data[f'v_{axis}'].iloc[:-n].copy()
        a_axis = data[f'a_{axis}'].iloc[:-n].copy()
        v_axis[np.abs(v_axis) < epsilon] = 0
        mask_a_axis = (np.abs(a_axis) <= 1000)
        v_axis = v_axis[mask_a_axis]
        f_axis_sim = data[f'f_{axis}_sim'].iloc[:-n][mask_a_axis]
        curr_axis = -data[f'curr_{axis}'].iloc[:-n][mask_a_axis]
        mrr = data['materialremoved_sim'].iloc[:-n][mask_a_axis]

        # Calculate medians
        median_v_axis = np.median(np.abs(v_axis))
        median_mrr = np.median(np.abs(mrr))
        median_f_axis_sim = np.median(np.abs(f_axis_sim))

        # Get color and label for the current file
        color, label = file_colors[file]

        # Plot data for v_axis > 0
        mask_gt_zero = v_axis > 0
        axs[0].scatter(f_axis_sim[mask_gt_zero], curr_axis[mask_gt_zero], c=color, s=2, alpha=0.5, label=label)
        axs[0].set_xlabel(f'f_{axis}_sim')
        axs[0].set_ylabel(f'curr_{axis}')
        axs[0].set_title(f'Plot for v_{axis} > 0')

        # Fit linear function for v_axis > 0
        if len(f_axis_sim[mask_gt_zero]) > 1 and len(curr_axis[mask_gt_zero]) > 1:
            popt_gt_zero, _ = curve_fit(linear_func, f_axis_sim[mask_gt_zero], curr_axis[mask_gt_zero])
            a_gt_zero, b_gt_zero = popt_gt_zero
            axs[0].plot(f_axis_sim[mask_gt_zero], linear_func(f_axis_sim[mask_gt_zero], *popt_gt_zero), color=color, linestyle='--')
            results.append({'File': file, 'Range': f'v_{axis} > 0', 'a': a_gt_zero, 'b': b_gt_zero,
                            f'Median |v_{axis}|': median_v_axis, 'Median |mrr|': median_mrr,
                            f'Median |f_{axis}_sim|': median_f_axis_sim})
            parameters[(file, f'v_{axis} > 0')] = (a_gt_zero, b_gt_zero, median_v_axis, median_mrr, median_f_axis_sim)

        # Plot data for v_axis < 0
        mask_lt_zero = v_axis < 0
        axs[1].scatter(f_axis_sim[mask_lt_zero], curr_axis[mask_lt_zero], color=color, s=2, alpha=0.5, label=label)
        axs[1].set_xlabel(f'f_{axis}_sim')
        axs[1].set_ylabel(f'curr_{axis}')
        axs[1].set_title(f'Plot for v_{axis} < 0')

        # Fit linear function for v_axis < 0
        if len(f_axis_sim[mask_lt_zero]) > 1 and len(curr_axis[mask_lt_zero]) > 1:
            popt_lt_zero, _ = curve_fit(linear_func, f_axis_sim[mask_lt_zero], curr_axis[mask_lt_zero])
            a_lt_zero, b_lt_zero = popt_lt_zero
            axs[1].plot(f_axis_sim[mask_lt_zero], linear_func(f_axis_sim[mask_lt_zero], *popt_lt_zero), color=color, linestyle='--')
            results.append({'File': file, 'Range': f'v_{axis} < 0', 'a': a_lt_zero, 'b': b_lt_zero,
                            f'Median |v_{axis}|': median_v_axis, 'Median |mrr|': median_mrr,
                            f'Median |f_{axis}_sim|': median_f_axis_sim})
            parameters[(file, f'v_{axis} < 0')] = (a_lt_zero, b_lt_zero, median_v_axis, median_mrr, median_f_axis_sim)

        # Plot data for v_axis == 0
        mask_eq_zero = v_axis == 0
        axs[2].scatter(f_axis_sim[mask_eq_zero], curr_axis[mask_eq_zero], c=color, s=2, alpha=0.5, label=label)
        axs[2].set_xlabel(f'f_{axis}_sim')
        axs[2].set_ylabel(f'curr_{axis}')
        axs[2].set_title(f'Plot for v_{axis} == 0')

        # Fit linear function for v_axis == 0
        if len(f_axis_sim[mask_eq_zero]) > 1 and len(curr_axis[mask_eq_zero]) > 1:
            popt_eq_zero, _ = curve_fit(linear_func, f_axis_sim[mask_eq_zero], curr_axis[mask_eq_zero])
            a_eq_zero, b_eq_zero = popt_eq_zero
            axs[2].plot(f_axis_sim[mask_eq_zero], linear_func(f_axis_sim[mask_eq_zero], *popt_eq_zero), color=color, linestyle='--')
            results.append({'File': file, 'Range': f'v_{axis} == 0', 'a': a_eq_zero, 'b': b_eq_zero,
                            f'Median |v_{axis}|': median_v_axis, 'Median |mrr|': median_mrr,
                            f'Median |f_{axis}_sim|': median_f_axis_sim})
            parameters[(file, f'v_{axis} == 0')] = (a_eq_zero, b_eq_zero, median_v_axis, median_mrr, median_f_axis_sim)

    # Gemeinsame Legende nur einmal unterhalb anzeigen
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Duplikate entfernen

    fig.legend(by_label.values(), by_label.keys(), loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=10, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)  # Platz für Legende unten
    plt.show()

    # Print parameters for sign(v_axis) = 1 and sign(v_axis) = -1
    print(f"Parameters for sign(v_{axis}) = 1:")
    for key, value in parameters.items():
        if f"v_{axis} > 0" in key[1]:
            print(f"{key[0]}, {key[1]}: a = {value[0]}, b = {value[1]}, Median |v_{axis}| = {value[2]}, Median |mrr| = {value[3]}, Median |f_{axis}_sim| = {value[4]}")

    print(f"\nParameters for sign(v_{axis}) = -1:")
    for key, value in parameters.items():
        if f"v_{axis} < 0" in key[1]:
            print(f"{key[0]}, {key[1]}: a = {value[0]}, b = {value[1]}, Median |v_{axis}| = {value[2]}, Median |mrr| = {value[3]}, Median |f_{axis}_sim| = {value[4]}")

    # Create additional plots for v_x = 1 and v_x = -1
    fig_files, axs_files = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig_vx, axs_vx = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig_mrr, axs_mrr = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    f_x_range = np.linspace(-500, 500, 500)

    # Get the maximum median values for normalization
    max_median_v = np.max([v[2] for v in parameters.values()])
    max_median_mrr = np.max([v[3] for v in parameters.values()])

    for key, value in parameters.items():
        file, range_label = key
        a, b, median_v_axis, median_mrr, median_f_axis_sim = value
        color, label = file_colors[file]

        # Calculate y values of the linear function for the v_x range
        y_fit = linear_func(f_x_range, a, b)

        # Plot in fig_files: color by file
        if 'v_' in range_label:
            if '> 0' in range_label:
                axs_files[0].plot(f_x_range, y_fit, color=color, label=label)
            elif '< 0' in range_label:
                axs_files[1].plot(f_x_range, y_fit, color=color, label=label)

            # Plot in fig_vx: color gradient based on median_v_axis
            norm_v = mcolors.Normalize(vmin=0, vmax=max_median_v)
            color_v = plt.cm.viridis(norm_v(median_v_axis))
            if '> 0' in range_label:
                axs_vx[0].plot(f_x_range, y_fit, color=color_v)
            elif '< 0' in range_label:
                axs_vx[1].plot(f_x_range, y_fit, color=color_v)

            # Plot in fig_mrr: color gradient based on median_mrr
            norm_mrr = mcolors.Normalize(vmin=0, vmax=max_median_mrr)
            color_mrr = plt.cm.plasma(norm_mrr(median_mrr))
            if '> 0' in range_label:
                axs_mrr[0].plot(f_x_range, y_fit, color=color_mrr)
            elif '< 0' in range_label:
                axs_mrr[1].plot(f_x_range, y_fit, color=color_mrr)

    # Create a ScalarMappable for each colormap to add to the legend
    sm_v = ScalarMappable(cmap='viridis', norm=norm_v)
    sm_v.set_array([])
    sm_mrr = ScalarMappable(cmap='plasma', norm=norm_mrr)
    sm_mrr.set_array([])

    # Set labels and titles for the additional plots
    titles = [
        'Fits nach Datei (v_axis >0 / <0)',
        'Fits mit Farbverlauf ~ Median |v_axis|',
        'Fits mit Farbverlauf ~ Median |mrr|'
    ]

    for axs_set, title in zip([axs_files, axs_vx, axs_mrr], titles):
        axs_set[0].set_title(f'{title} - v_axis > 0')
        axs_set[1].set_title(f'{title} - v_axis < 0')
        for ax in axs_set:
            ax.set_xlabel('f_axis_sim')
            ax.set_ylabel('curr_axis')
            ax.grid(True)

        # Add colorbar legend for the color gradients
        if axs_set is axs_vx:
            cbar_v = fig_vx.colorbar(sm_v, ax=axs_vx[1], orientation='vertical')
            cbar_v.set_label('Median |v_axis|')
        elif axs_set is axs_mrr:
            cbar_mrr = fig_mrr.colorbar(sm_mrr, ax=axs_mrr[1], orientation='vertical')
            cbar_mrr.set_label('Median |mrr|')

        # Optional legend only for the first figure (file-based)
        if axs_set is axs_files:
            handles, labels = axs_set[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axs_set[0].legend(by_label.values(), by_label.keys(), loc='best', fontsize=8)

    plt.tight_layout()
    plt.show()

    fig_params, axs_params = plt.subplots(1, 3, figsize=(21, 6), dpi=300)
    vx_ranges = [f'v_{axis} > 0', f'v_{axis} < 0', f'v_{axis} == 0']

    for i, vx_range in enumerate(vx_ranges):
        al_plate_as = []
        al_plate_bs = []
        s_plate_as = []
        s_plate_bs = []

        al_gear_as = []
        al_gear_bs = []
        s_gear_as = []
        s_gear_bs = []

        for (file, range_key), (a, b, *_) in parameters.items():
            if range_key != vx_range:
                continue
            # Kategorisierung nach 'AL' oder 'S' am Dateinamenanfang
            if file.startswith("AL") and "Plate" in file:
                al_plate_as.append(a)
                al_plate_bs.append(b)
            elif file.startswith("S") and "Plate" in file:
                s_plate_as.append(a)
                s_plate_bs.append(b)
            elif file.startswith("AL") and "Gear" in file:
                al_gear_as.append(a)
                al_gear_bs.append(b)
            elif file.startswith("S") and "Gear" in file:
                s_gear_as.append(a)
                s_gear_bs.append(b)

        axs_params[i].scatter(al_plate_as, al_plate_bs, color='blue', label='AL Plate', alpha=0.7, edgecolor='k')
        axs_params[i].scatter(al_gear_as, al_gear_bs, color='green', label='AL Gear', alpha=0.7, edgecolor='k')
        axs_params[i].scatter(s_plate_as, s_plate_bs, color='red', label='S Plate', alpha=0.7, edgecolor='k')
        axs_params[i].scatter(s_gear_as, s_gear_bs, color='orange', label='S Gear', alpha=0.7, edgecolor='k')

        axs_params[i].set_title(f"a vs b for {vx_range}")
        axs_params[i].set_xlabel('a (Steigung)')
        axs_params[i].set_ylabel('b (Achsenabschnitt)')
        axs_params[i].grid(True)

        # Falls alle Punkte fehlen, füge Dummy-Text hinzu
        if len(al_plate_as) + len(al_gear_as) + len(s_gear_as) + len(s_gear_as) == 0:
            axs_params[i].text(0.5, 0.5, 'Keine Daten', ha='center', va='center', fontsize=14, color='gray',
                               transform=axs_params[i].transAxes)

    # Gemeinsame Legende nur einmal
    handles, labels = axs_params[0].get_legend_handles_labels()
    fig_params.legend(handles, labels, loc='lower center', ncol=2, fontsize=12, frameon=False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


    # Zunächst aus parameters dict je sign(v_axis) (>, <, == 0) die Werte sammeln
    a_vals = []
    b_vals = []
    median_mrr_vals = []
    median_abs_v_vals = []
    median_f_axis_vals = []
    median_f_sp_vals = []

    # Iteriere über alle gesammelten Parameter für die aktuelle Achse
    for (file, v_range), (a_param, b_param, median_v, median_mrr, median_f_axis) in parameters.items():
        if axis in v_range:
            # Lade nochmal f_sp median für die Datei (einfach aus der CSV)
            data = pd.read_csv(f'{path_data}/{file}')
            mask = np.abs(data[f'a_{axis}']) <= 1000
            median_f_sp = np.median(np.abs(data['f_sp_sim'].iloc[:-n][mask])) if 'f_sp_sim' in data.columns else np.nan

            a_vals.append(a_param)
            b_vals.append(b_param)
            median_mrr_vals.append(median_mrr)
            median_abs_v_vals.append(median_v)
            median_f_axis_vals.append(median_f_axis)
            median_f_sp_vals.append(median_f_sp)

    # Scatterplots erstellen
    fig2, axs2 = plt.subplots(2, 4, figsize=(24, 10), dpi=300)
    axs2 = axs2.flatten()

    axs2[0].scatter(a_vals, median_mrr_vals, c='blue', alpha=0.7)
    axs2[0].set_xlabel('a')
    axs2[0].set_ylabel('Median MRR')
    axs2[0].set_title(f'a vs Median MRR (axis={axis})')

    axs2[1].scatter(b_vals, median_mrr_vals, c='green', alpha=0.7)
    axs2[1].set_xlabel('b')
    axs2[1].set_ylabel('Median MRR')
    axs2[1].set_title(f'b vs Median MRR (axis={axis})')

    axs2[2].scatter(a_vals, median_abs_v_vals, c='blue', alpha=0.7)
    axs2[2].set_xlabel('a')
    axs2[2].set_ylabel(f'Median |v_{axis}|')
    axs2[2].set_title(f'a vs Median |v_{axis}|')

    axs2[3].scatter(b_vals, median_abs_v_vals, c='green', alpha=0.7)
    axs2[3].set_xlabel('b')
    axs2[3].set_ylabel(f'Median |v_{axis}|')
    axs2[3].set_title(f'b vs Median |v_{axis}|')

    axs2[4].scatter(a_vals, median_f_axis_vals, c='blue', alpha=0.7)
    axs2[4].set_xlabel('a')
    axs2[4].set_ylabel(f'Median |f_{axis}_sim|')
    axs2[4].set_title(f'a vs Median |f_{axis}_sim|')

    axs2[5].scatter(b_vals, median_f_axis_vals, c='green', alpha=0.7)
    axs2[5].set_xlabel('b')
    axs2[5].set_ylabel(f'Median |f_{axis}_sim|')
    axs2[5].set_title(f'b vs Median |f_{axis}_sim|')

    axs2[6].scatter(a_vals, median_f_sp_vals, c='blue', alpha=0.7)
    axs2[6].set_xlabel('a')
    axs2[6].set_ylabel('Median |f_sp|')
    axs2[6].set_title(f'a vs Median |f_sp|')

    axs2[7].scatter(b_vals, median_f_sp_vals, c='green', alpha=0.7)
    axs2[7].set_xlabel('b')
    axs2[7].set_ylabel('Median |f_sp|')
    axs2[7].set_title(f'b vs Median |f_sp|')

    plt.tight_layout()
    plt.show()

    # Calculate differences between positive and negative ranges for each file
    differences = {}

    for file in set([key[0] for key in parameters.keys()]):
        pos_key = (file, 'v_x > 0')
        neg_key = (file, 'v_x < 0')

        if pos_key in parameters and neg_key in parameters:
            a_pos, b_pos, _, _, _ = parameters[pos_key]
            a_neg, b_neg, _, _, _ = parameters[neg_key]

            differences[file] = {
                'Δa': 2*(a_pos - a_neg)/(a_pos + a_pos),
                'Δb': 2*(np.abs(b_pos) - np.abs(b_neg))/ (np.abs(b_pos) + np.abs(b_neg))# b_neg ist negativ
            }

    differences_df = pd.DataFrame(differences).T

    # Plotting the differences
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for Δa
    ax[0].bar(differences_df.index, differences_df['Δa'], color='skyblue')
    ax[0].set_title('Relative Differences in parameter a')
    ax[0].set_ylabel('Δa')
    ax[0].set_xticklabels(differences_df.index, rotation=45, ha='right')

    # Plot for Δb
    ax[1].bar(differences_df.index, differences_df['Δb'], color='lightgreen')
    ax[1].set_title('Relative Differences in parameter abs(b)')
    ax[1].set_ylabel('Δb')
    ax[1].set_xticklabels(differences_df.index, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()