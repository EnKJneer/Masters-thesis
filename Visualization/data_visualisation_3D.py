import os
from collections import deque
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sign_hold(v, eps = 1e0, n=3):
    # Initialisierung des Arrays z mit Nullen
    z = np.zeros(len(v))
    h_init = np.ones(n)

    assert n > 1

    # Initialisierung des FiFo h mit Länge 5 und Initialwerten 0
    h = deque(h_init, maxlen=n)

    # Berechnung von z
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])

        if i >= n-1:  # Da wir ab dem 5. Element starten wollen
            # Berechne zi als Vorzeichen der Summe
            z[i] = np.sign(sum(h))

    return z


def plot_3d_with_color(
    x_values, y_values, z_values, color_values,
    material_values,  # Neue Spalte mit Materialnamen
    title='3D Plot', filename='3d_plot.png',
    dpi=300, xlabel='X-Achse', ylabel='Y-Achse', zlabel='Z-Achse',
    colorbar_label='Farbwerte', path='Plots'
):
    # KIT-Farben
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_orange = "#FFC000"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_magenta = "#A3107C"

    material_colormaps = {
        'AL_2007_T4': 'bwr',
        'S235JR': 'seismic',
        # Weitere Materialien und Colormaps nach Bedarf
    }

    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Für jedes Material separat plotten
    for mat in np.unique(material_values):
        mat_mask = (material_values == mat)
        sc = ax.scatter(
            x_values[mat_mask],
            y_values[mat_mask],
            z_values[mat_mask],
            c=color_values[mat_mask],
            cmap=material_colormaps.get(mat, 'viridis'),
            s=1, edgecolor='none', label=mat
        )

    # DIN 461: Achsenbeschriftung
    ax.set_xlabel(xlabel, color=kit_dark_blue, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, color=kit_dark_blue, fontsize=12, labelpad=10)
    ax.set_zlabel(zlabel, color=kit_dark_blue, fontsize=12, labelpad=10)
    ax.set_title(title, color=kit_dark_blue, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5, linestyle='-')
    ax.tick_params(axis='x', colors=kit_dark_blue, direction='inout', length=6)
    ax.tick_params(axis='y', colors=kit_dark_blue, direction='inout', length=6)
    ax.tick_params(axis='z', colors=kit_dark_blue, direction='inout', length=6)

    # Farbskala (nur eine, da alle Colormaps auf z_x basieren)
    sc = ax.scatter([], [], [], c=[], cmap=material_colormaps[np.unique(material_values)[0]], vmin=color_values.min(), vmax=color_values.max())
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(colorbar_label, color=kit_dark_blue, fontsize=12)

    #ax.legend()
    plt.show()
'''    # Speicherpfad erstellen und Plot speichern
    plot_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)'''

path_data = '..\\DataSets\\DataSimulated'
files = os.listdir(path_data)
materials = ['AL_2007_T4', 'S235JR']  # Liste der Materialien
df = []

for file in files:
    # Prüfe, mit welchem Material die Datei beginnt
    for mat in materials:
        if file.startswith(mat):
            data = pd.read_csv(f'{path_data}/{file}')
            data['z_x'] = sign_hold(data['v_x'])
            data['material'] = mat  # Materialname in der neuen Spalte speichern
            df.append(data)
            break  # Nur das erste passende Material verwenden

data = pd.concat(df)
mask = (abs(data['v_x']) < 1)
data = data[mask]

plot_3d_with_color(
    data['v_x'], -data['f_x_sim'], data['curr_x'], data['z_x'],
    material_values=data['material'],  # Material-Spalte übergeben
    xlabel='$v$ in m/s', ylabel='$F_{sim}$ in N', zlabel='$I$ in A',
    title='3D-Trajektorie mit Kraft (Colormap pro Material)',
    filename='3d_trajectory_by_material.png'
)
