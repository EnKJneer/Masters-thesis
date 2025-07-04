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
    'AL_2007_T4_Plate_Normal_1.csv', 'AL_2007_T4_Gear_Normal_1.csv',
    'AL_2007_T4_Plate_SF_1.csv', 'AL_2007_T4_Gear_SF_1.csv',
    'AL_2007_T4_Plate_Depth_1.csv', 'AL_2007_T4_Gear_Depth_1.csv',
    'AL_2007_T4_Plate_Depth_2.csv', 'AL_2007_T4_Gear_Depth_2.csv',
    'AL_2007_T4_Plate_Normal_2.csv', 'AL_2007_T4_Gear_Normal_2.csv',
    'AL_2007_T4_Plate_SF_2.csv', 'AL_2007_T4_Gear_SF_2.csv',
    'AL_2007_T4_Plate_Depth_3.csv', 'AL_2007_T4_Gear_Depth_3.csv',
    'S235JR_Gear_Normal_1.csv', 'S235JR_Plate_Normal_1.csv',
    'S235JR_Plate_SF_1.csv', 'S235JR_Gear_SF_1.csv',
    'S235JR_Plate_Depth_1.csv', 'S235JR_Gear_Depth_1.csv',
    'S235JR_Plate_Depth_2.csv', 'S235JR_Gear_Depth_2.csv',
    'AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv',
    'AL_2007_T4_Plate_SF_3.csv', 'AL_2007_T4_Gear_SF_3.csv',
    'S235JR_Plate_Depth_3.csv', 'S235JR_Gear_Depth_3.csv',
    'S235JR_Plate_Normal_2.csv', 'S235JR_Gear_Normal_2.csv',
    'S235JR_Plate_SF_2.csv', 'S235JR_Gear_SF_2.csv',
    'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv',
    'S235JR_Plate_SF_3.csv', 'S235JR_Gear_SF_3.csv'
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
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    for file in files:
        data = pd.read_csv(f'{path_data}/{file}')

        epsilon = 1e-1
        a_axis = data[f'a_{axis}'].iloc[:-n].copy()
        v_axis = data[f'v_{axis}'].iloc[:-n].copy()
        f_axis = data[f'f_{axis}_sim'].iloc[:-n].copy()
        curr_axis = -data[f'curr_{axis}'].iloc[:-n].copy()
        mrr = data['materialremoved_sim'].iloc[:-n].copy()
        time = data.index[:-n].copy()

        # Maske, um evtl. Ausreißer zu entfernen (optional, wie in Original)
        mask = (np.abs(v_axis) <= 1000) #& (np.abs(v_axis) <= epsilon)  # kann angepasst werden
        v_axis = v_axis[mask]
        f_axis = f_axis[mask]
        curr_axis = curr_axis[mask]
        z = sign_hold(v_axis)

        '''        # Figur und Achse erstellen
        fig2, axs2 = plt.subplots(figsize=(10, 6))

        # Streudiagramm für curr_axis auf der primären y-Achse
        axs2.scatter(time, curr_axis, label='curr_axis', s=2, color='blue')
        axs2.set_xlabel('Time')
        axs2.set_ylabel('curr_axis', color='blue')
        axs2.tick_params(axis='y', labelcolor='blue')

        # Zweite y-Achse für v_axis erstellen
        axs2_twin = axs2.twinx()

        # Linienplot für v_axis auf der sekundären y-Achse
        axs2_twin.scatter(time, v_axis, label='v_axis', s=2, color='red')
        axs2_twin.set_ylabel('v_axis', color='red')
        axs2_twin.tick_params(axis='y', labelcolor='red')

        # Titel hinzufügen
        axs2.set_title('Plot of curr_axis and v_axis over time')

        # Legenden hinzufügen
        lines, labels = axs2.get_legend_handles_labels()
        lines2, labels2 = axs2_twin.get_legend_handles_labels()
        axs2.legend(lines + lines2, labels + labels2, loc='upper right')

        fig2.show()'''

        d = pd.Series(v_axis)
        v_axis = np.array(d.rolling(50).mean())
        d = pd.Series(curr_axis)
        curr_axis = np.array(d.rolling(50).mean().fillna(0))

        color, label = file_colors.get(file, ('black', file))

        # Scatterplot v vs curr
        axs[0].scatter(v_axis, curr_axis, c=color, s=2, alpha=0.5, label=label)
        axs[0].set_xlabel(f'v_{axis}')
        axs[0].set_ylabel(f'curr_{axis}')
        axs[0].set_title(f'Scatterplot curr vs v for axis {axis}')
        xlimit = 10
        axs[0].set_xlim(max(-xlimit, min(v_axis)*1.1), min(xlimit, max(v_axis)*1.1))
        ylimit = 3
        axs[0].set_ylim(max(-ylimit, min(curr_axis)*1.1), min(ylimit, max(curr_axis)*1.1))

        # Fit der linearen Funktion curr ~ a*v + b
        if len(v_axis) > 1 and len(curr_axis) > 1:
            popt, _ = curve_fit(linear_func, f_axis, curr_axis)
            a, b = popt

            # Approximation berechnen
            curr_approx = linear_func(f_axis, a, 0)

            # Plot Fitlinie
            sort_idx = np.argsort(v_axis)
            #axs[0].plot(v_axis.iloc[sort_idx], curr_approx[sort_idx], color=color, linestyle='--')

            # Differenz berechnen
            diff = curr_axis - curr_approx

            # Differenz vs v plotten im 2. Subplot
            axs[1].plot(v_axis, diff, color='gray', alpha=0.5, linewidth=0.5)
            axs[1].scatter(v_axis, diff, c=color, s=5, alpha=0.8)
            #axs[1].plot(v_axis, diff, c=color, alpha=0.25, label=label)
            axs[1].set_xlabel(f'v_{axis}')
            axs[1].set_ylabel('curr - curr_approx')
            axs[1].set_title(f'Difference curr - approx vs v for axis {axis}')
            axs[1].set_xlim(max(-xlimit, min(v_axis)*1.1), min(xlimit, max(v_axis)*1.1))
            axs[1].set_ylim(max(-ylimit, min(diff)*1.1), min(ylimit, max(diff)*1.1))


    # Legende nur einmal anzeigen
    handles, labels = axs[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center',
               bbox_to_anchor=(0.5, 0.0), ncol=5, fontsize=10, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)  # Platz für Legende unten
    plt.show()
