import os
from collections import deque
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
def sign_hold(v, eps = 1e0):
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
def plot_3d_with_color(
    x_values, y_values, z_values, color_values,
    title='3D Plot', filename='3d_plot.png',
    dpi=300, xlabel='X-Achse', ylabel='Y-Achse', zlabel='Z-Achse',
    colorbar_label='Farbwerte', path='Plots'
):
    """
    Erstellt einen DIN 461-inspirierten 3D-Plot mit Farbskala.
    """
    # KIT-Farben
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_orange = "#FFC000"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_magenta = "#A3107C"

    # DIN 461: Figur mit Seitenverhältnis ca. 3:2
    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # 3D-Plot: Punkte mit/ohne Farbskala
    if color_values is not None:
        sc = ax.scatter(
            x_values, y_values, z_values,
            c=color_values, cmap='viridis', s=1, edgecolor='none'
        )
    else:
        sc = ax.scatter(
            x_values, y_values, z_values,
            color = kit_blue,
            s=1, edgecolor='none'
        )

    # DIN 461: Achsenbeschriftung mit Text (Pfeile sind in 3D schwierig, daher nur Text)
    ax.set_xlabel(xlabel, color=kit_dark_blue, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, color=kit_dark_blue, fontsize=12, labelpad=10)
    ax.set_zlabel(zlabel, color=kit_dark_blue, fontsize=12, labelpad=10)

    # Farbskala nur, wenn color_values nicht None
    if color_values is not None:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label(colorbar_label, color=kit_dark_blue, fontsize=12)

    # Titel
    ax.set_title(title, color=kit_dark_blue, fontsize=14, fontweight='bold', pad=20)

    # Grid (optional, aber DIN 461-konform)
    ax.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5, linestyle='-')

    # Achsen-Ticks und -Farben
    ax.tick_params(axis='x', colors=kit_dark_blue, direction='inout', length=6)
    ax.tick_params(axis='y', colors=kit_dark_blue, direction='inout', length=6)
    ax.tick_params(axis='z', colors=kit_dark_blue, direction='inout', length=6)

    plt.show()
'''    # Speicherpfad erstellen und Plot speichern
    plot_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)'''

# Beispielaufruf (mit deinen Daten)
path_data = '..\\DataSets\\DataSimulated'
files = os.listdir(path_data)
files = ['AL_2007_T4_Plate_Depth.csv']
for file in files:
    data = pd.read_csv(f'{path_data}/{file}')
    data['z_x'] = sign_hold(data['v_x'])
    mask = (abs(data['v_x']) < 1)
    #data = data[mask]
    plot_3d_with_color(
        data['z_x'], data['curr_x'], data['f_x'],  None,
        xlabel='$z$', ylabel='$I$ in A', zlabel='$F$ in N',
        title='3D-Trajektorie mit Kraft', filename='3d_trajectory.png'
    )
