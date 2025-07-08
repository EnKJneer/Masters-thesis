import os
from collections import deque

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
    #'AL_2007_T4_Plate_Depth_2.csv', 'AL_2007_T4_Gear_Depth_2.csv',
    'AL_2007_T4_Plate_Normal_2.csv', 'AL_2007_T4_Gear_Normal_2.csv',
    #'AL_2007_T4_Plate_SF_2.csv', 'AL_2007_T4_Gear_SF_2.csv',
    #'AL_2007_T4_Plate_Depth_3.csv', 'AL_2007_T4_Gear_Depth_3.csv',
    #'S235JR_Gear_Normal_1.csv', 'S235JR_Plate_Normal_1.csv',
    #'S235JR_Plate_SF_1.csv', 'S235JR_Gear_SF_1.csv',
    #'S235JR_Plate_Depth_1.csv', 'S235JR_Gear_Depth_1.csv',
    #'S235JR_Plate_Depth_2.csv', 'S235JR_Gear_Depth_2.csv',
    #'AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv',
    #'AL_2007_T4_Plate_SF_3.csv', 'AL_2007_T4_Gear_SF_3.csv',
    #'S235JR_Plate_Depth_3.csv', 'S235JR_Gear_Depth_3.csv',
    'S235JR_Plate_Normal_2.csv', 'S235JR_Gear_Normal_2.csv',
    #'S235JR_Plate_SF_2.csv', 'S235JR_Gear_SF_2.csv',
    #'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv',
    #'S235JR_Plate_SF_3.csv', 'S235JR_Gear_SF_3.csv'
]
n = 25
axes = ['x'] #, 'y'

def sign_hold(v, eps = 1e-1):
    # Initialisierung des Arrays z mit Nullen
    z = np.zeros(len(v))

    # Initialisierung des FiFo h mit Länge 5 und Initialwerten 0
    h = deque([1, 1, 1, 1, 1], maxlen=5)

    # Berechnung von z
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])

        if i >= 4:  # Da wir ab dem 5. Element starten wollen
            # Berechne zi als Vorzeichen der Summe
            z[i] = np.sign(sum(h))

    return z

for axis in axes:
    for file in files:
        data = pd.read_csv(f'{path_data}/{file}')

        epsilon = 1e-1
        a_axis = data[f'a_{axis}'].iloc[:-n].copy()
        v_axis = data[f'v_{axis}'].iloc[:-n].copy()
        f_axis = data[f'f_{axis}_sim'].iloc[:-n].copy()
        f_y = data[f'f_y_sim'].iloc[:-n].copy()
        curr_axis = -data[f'curr_{axis}'].iloc[:-n].copy()
        mrr = data['materialremoved_sim'].iloc[:-n].copy()
        time = data.index[:-n].copy()

        # Maske, um evtl. Ausreißer zu entfernen (optional, wie in Original)
        mask = (np.abs(v_axis) <= 1000) & (np.abs(a_axis) <= 10)#& (np.abs(v_axis) <= epsilon)  # kann angepasst werden
        v_axis = v_axis[mask].reset_index(drop=True)
        f_axis = f_axis[mask].reset_index(drop=True)
        f_y = f_y[mask].reset_index(drop=True)
        a_axis = a_axis[mask].reset_index(drop=True)
        curr_axis = curr_axis[mask].reset_index(drop=True)
        z = sign_hold(v_axis)

        d = pd.Series(v_axis)
        v_axis = np.array(d.rolling(10).mean())
        d = pd.Series(curr_axis)
        curr_axis = np.array(d.rolling(10).mean().fillna(0))

        color, label = file_colors.get(file, ('black', file))

        popt, _ = curve_fit(linear_func, f_axis, curr_axis)
        a, b = popt

        # Approximation berechnen
        curr_approx = linear_func(f_axis, a, 0)

        # Plot Fitlinie
        sort_idx = np.argsort(v_axis)

        # Differenz berechnen
        diff = curr_axis - curr_approx

        y_datas = [("curr_axis", curr_axis)] #, ("diff", diff)

        color_value = ("a_axis", a_axis)
        for y_data in y_datas:

            fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
            fig.suptitle(file)

            # Scatterplot v vs curr
            axs[0].scatter(v_axis, y_data[1], c=color_value[1], s=2, alpha=0.5, label=label)
            axs[0].set_xlabel(f'v_{axis}')
            axs[0].set_ylabel(y_data[0])
            axs[0].set_title(f'Scatterplot {y_data[0]} vs v for axis {axis}')
            xlimit = 2
            #axs[0].set_xlim(max(-xlimit, min(v_axis)*1.1), min(xlimit, max(v_axis)*1.1))
            ylimit = 3
            #axs[0].set_ylim(max(-ylimit, min(curr_axis)*1.1), min(ylimit, max(curr_axis)*1.1))

            # Pfeile zwischen aufeinanderfolgenden Punkten
            dx = []
            dy = []
            for i in range(len(v_axis) - 1):
                dx.append(v_axis[i + 1] - v_axis[i])
                dy.append(y_data[1][i + 1] - y_data[1][i])
            axs[0].quiver(v_axis[:-1], y_data[1][:-1], dx, dy, angles='xy', scale_units='xy', scale=1, width=0.005,
                          color='gray',
                          alpha=0.5)

            axs[1].quiver(v_axis[:-1], y_data[1][:-1], dx, dy, angles='xy', scale_units='xy', scale=1, width=0.005,
                          color='gray',
                          alpha=0.5)

            # Differenz vs v plotten im 2. Subplot
            axs[1].plot(v_axis, y_data[1], color='gray', alpha=0.5, linewidth=0.5)
            axs[1].scatter(v_axis, y_data[1], c=color_value[1], s=5, alpha=0.8)
            #axs[1].plot(v_axis, diff, c=color, alpha=0.25, label=label)
            axs[1].set_xlabel(f'v_{axis}')
            axs[1].set_ylabel(y_data[0])
            axs[1].set_title(f'{y_data[0]} vs v for axis {axis}')
            axs[1].set_xlim(max(-xlimit, min(v_axis)*1.1), min(xlimit, max(v_axis)*1.1))
            axs[1].set_ylim(max(-ylimit, min(y_data[1])*1.1), min(ylimit, max(y_data[1])*1.1))

            # Normalize time
            norm = mcolors.Normalize(vmin=np.min(color_value[1]), vmax=np.max(color_value[1]))
            sm = plt.cm.ScalarMappable(norm=norm)
            sm.set_array([])

            # Add colorbar for the derivative
            plt.colorbar(sm, ax=axs[1], label=color_value[0])

            # Legende nur einmal anzeigen
            handles, labels = axs[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            #fig.legend(by_label.values(), by_label.keys(), loc='lower center',
            #           bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=10, frameon=False)

            plt.tight_layout()
            #plt.subplots_adjust(bottom=0.4)  # Platz für Legende unten
            plt.show()



