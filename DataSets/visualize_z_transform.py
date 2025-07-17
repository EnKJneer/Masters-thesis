import os
from collections import deque

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import lstsq
from scipy.signal import dlti, freqresp, dfreqresp
from scipy.optimize import curve_fit
from scipy import signal
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

        # Beispiel: Eingang = v_x, Ausgang = a_x
        u = f_axis.values
        y = curr_axis.values

        # Modellordnung festlegen (z. B. ARX mit na=2, nb=2)
        na = 2  # Ordnung Ausgang
        nb = 2  # Ordnung Eingang
        N = len(y)
        dt = 0.02

        # Regressionsmatrix X aufbauen
        X = []
        Y = []

        for k in range(max(na, nb), N):
            row = []
            # Negative y-Werte (autoregressiver Teil)
            row += [-y[k - i] for i in range(1, na + 1)]
            # Positive u-Werte (input-Teil)
            row += [u[k - i] for i in range(1, nb + 1)]
            X.append(row)
            Y.append(y[k])

        X = np.array(X)
        Y = np.array(Y)

        # Least Squares Schätzung
        theta, _, _, _ = lstsq(X, Y, rcond=None)
        a_est = theta[:na]
        b_est = theta[na:]

        print("a-Koeffizienten:", a_est)
        print("b-Koeffizienten:", b_est)

        # Erzeuge z-Transferfunktion
        b = np.concatenate([[0], b_est])  # Delay durch z^-1
        a = np.concatenate([[1], a_est])  # Vorzeichen beachten

        system = dlti(b, a, dt=dt)
        w, h = dfreqresp(system)

        # Frequenz in Hz
        freqs = w / (2 * np.pi)

        plt.plot(freqs, np.abs(h))
        plt.title('Frequenzgang des geschätzten Systems')
        plt.xlabel('Frequenz [Hz]')
        plt.ylabel('|H(f)|')
        plt.grid(True)
        plt.show()
