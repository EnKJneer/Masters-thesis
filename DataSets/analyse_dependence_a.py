import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    'AL_2007_T4_Plate_Normal_2.csv', 'AL_2007_T4_Gear_Normal_2.csv',
    'AL_2007_T4_Plate_SF_2.csv', 'AL_2007_T4_Gear_SF_2.csv',
    #'AL_2007_T4_Plate_Depth_3.csv', 'AL_2007_T4_Gear_Depth_3.csv',
    #'S235JR_Gear_Normal_1.csv', 'S235JR_Plate_Normal_1.csv',
    #'S235JR_Plate_SF_1.csv', 'S235JR_Gear_SF_1.csv',
    #'S235JR_Plate_Depth_1.csv', 'S235JR_Gear_Depth_1.csv',
    'S235JR_Plate_Depth_2.csv', 'S235JR_Gear_Depth_2.csv',
    #'S235JR_Plate_Depth_3.csv', 'S235JR_Gear_Depth_3.csv',
    #'AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv',
    #'AL_2007_T4_Plate_SF_3.csv', 'AL_2007_T4_Gear_SF_3.csv',
    'S235JR_Gear_Normal_2.csv', 'S235JR_Plate_Normal_2.csv',
    'S235JR_Plate_SF_2.csv', 'S235JR_Gear_SF_2.csv',
    #'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv',
    #'S235JR_Plate_SF_3.csv', 'S235JR_Gear_SF_3.csv'
]
n = 25
axes = ['x', 'y']

for axis in axes:
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    for file in files:
        data = pd.read_csv(f'{path_data}/{file}')

        epsilon = 1e-3
        a_axis = data[f'a_{axis}'].iloc[:-n].copy()
        v_axis = data[f'v_{axis}'].iloc[:-n].copy()
        f_axis = data[f'f_{axis}_sim'].iloc[:-n].copy()
        curr_axis = -data[f'curr_{axis}'].iloc[:-n].copy()

        # Maske, um evtl. Ausreißer zu entfernen (optional, wie in Original)
        mask = (np.abs(v_axis) <= 1000) & (v_axis > -epsilon)
        a_axis = a_axis[mask]
        f_axis = f_axis[mask]
        curr_axis = curr_axis[mask]

        color, label = file_colors.get(file, ('black', file))

        d = pd.Series(a_axis)

        a_axis = np.array(d.rolling(10).mean().fillna(0))

        # Scatterplot v vs curr
        axs[0].scatter(a_axis, curr_axis, c=color, s=2, alpha=0.5, label=label)
        axs[0].set_xlabel(f'a_{axis}')
        axs[0].set_ylabel(f'curr_{axis}')
        axs[0].set_title(f'Scatterplot curr vs a for axis {axis}')
        #xlimit = 1000
        #axs[0].set_xlim(max(-xlimit, min(a_axis)*1.1), min(xlimit, max(a_axis)*1.1))
        #ylimit = 3
        #axs[0].set_ylim(max(-ylimit, min(curr_axis)*1.1), min(ylimit, max(curr_axis)*1.1))

        # Fit der linearen Funktion curr ~ a*v + b
        if len(a_axis) > 1 and len(curr_axis) > 1:
            popt, _ = curve_fit(linear_func, f_axis, curr_axis)
            a, b = popt

            # Approximation berechnen
            curr_approx = linear_func(f_axis, a, b)

            # Plot Fitlinie
            #sort_idx = np.argsort(v_axis)
            #axs[0].plot(v_axis.iloc[sort_idx], curr_approx[sort_idx], color=color, linestyle='--')

            # Differenz berechnen
            diff = curr_axis - curr_approx

            popt, _ = curve_fit(linear_func, a_axis, diff)
            a, b = popt

            # Approximation berechnen
            curr_approx = linear_func(a_axis, a, b)
            # Plot Fitlinie
            sort_idx = np.argsort(a_axis)
            axs[1].plot(a_axis, curr_approx, color=color, linestyle='--')

            # Differenz vs v plotten im 2. Subplot
            axs[1].scatter(a_axis, diff, c=color, s=2, alpha=0.5, label=label)
            axs[1].set_xlabel(f'a_{axis}')
            axs[1].set_ylabel('curr - curr_approx')
            axs[1].set_title(f'Difference curr - approx vs v for axis {axis}')
            #axs[1].set_xlim(max(-xlimit, min(v_axis)*1.1), min(xlimit, max(v_axis)*1.1))
            #axs[1].set_ylim(max(-ylimit, min(diff)*1.1), min(ylimit, max(diff)*1.1))

    # Legende nur einmal anzeigen
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=10, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)  # Platz für Legende unten
    plt.show()
