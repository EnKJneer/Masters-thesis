import os
from collections import deque
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sympy import zeros

matplotlib.use("TkAgg")

def sign_hold(v, eps=1e0, n = 3):
    e = np.zeros(n)
    z = np.zeros(len(v))
    h = deque(e, maxlen=n)
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])
        if i >= n-1:
            z[i] = np.sign(sum(h))
    return z

def hold(v, eps=1e-1, n = 7):
    e = np.zeros(n)
    z = np.zeros(len(v))
    h = deque(e, maxlen=n)
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])
        if i >= n-1:
            z[i] = np.mean(h)
    return z

def plot_time_series_3d(data, title, gradients, dpi=300, s=1,
                       col_name_x='time', label_axis_x='$t$ in s',
                       col_name_y='v_x', label_axis_y='Geschwindigkeit in m/s',
                       col_name_z='f_x', label_axis_z='Kraft in N',
                       label='Messwerte', path='Plots'):
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"

    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    line0 = ax.scatter(data[col_name_x], data[col_name_y], data[col_name_z], label=label, color=kit_blue, s=s)

    ax.set_xlabel(label_axis_x, color=kit_dark_blue, fontsize=12)
    ax.set_ylabel(label_axis_y, color=kit_dark_blue, fontsize=12)
    ax.set_zlabel(label_axis_z, color=kit_dark_blue, fontsize=12)

    ax.set_title(title, color=kit_dark_blue, fontsize=14, fontweight='bold', pad=20)

    #ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False,
    #          framealpha=1.0, facecolor='white', edgecolor=kit_dark_blue)

    '''    
    # Plot the gradients as arrows
    for i in range(0, len(data[col_name_x])):
        ax.quiver(data[col_name_x].iloc[i], data[col_name_y].iloc[i], data[col_name_z].iloc[i],
                  gradients[0][i], gradients[1][i], gradients[2][i],
                  color=kit_green, length=1.0, arrow_length_ratio=0.1)'''

    plt.show()

# Beispielaufruf der Funktion
material = 'S235JR'
geometry = 'Plate'
versions = ['SF', 'Depth', 'Normal']  # Beispielhafte Liste von Versionen
path_data = '../../DataSets/DataSimulated'

# Leerer DataFrame zur Sammlung aller Daten
all_data = pd.DataFrame()

for version in versions:
    file = f'{material}_{geometry}_{version}.csv'
    data = pd.read_csv(os.path.join(path_data, file))

    # Berechnungen für jede Version
    data['z_x'] = sign_hold(data['v_x'])
    data['t_x'] = hold(data['v_x'])

    # Daten zum gemeinsamen DataFrame hinzufügen
    all_data = pd.concat([all_data, data], ignore_index=True)

# Maskierung und Gradientberechnung für den gesamten Datensatz
mask = (abs(all_data['v_x']) < 1)

gradients = (
    np.gradient(all_data['curr_x'][mask]) * 0.1,
    np.gradient(all_data['t_x'][mask]) * 0.1,
    np.gradient(all_data['f_x'][mask]) * 0.1
)

# Plot für den gesamten Datensatz
plot_time_series_3d(all_data[mask], f'{material} {geometry} Combined Versions',
                   gradients,
                   col_name_x='curr_x', label_axis_x='$I$ in A',
                   col_name_y='v_x', label_axis_y='$v$',
                   col_name_z='f_x_sim', label_axis_z='$F$ in N')
