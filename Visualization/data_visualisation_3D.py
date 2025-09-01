import os
from collections import deque
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sign_hold(v, eps=1e0, n=3):
    z = np.zeros(len(v))
    h_init = np.ones(n)
    assert n > 1
    h = deque(h_init, maxlen=n)
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])
        if i >= n-1:
            z[i] = np.sign(sum(h))
    return z

def plot_3d_with_color(
    x_values, y_values, z_values, color_values,
    material_values,
    u_grad, v_grad, w_grad,  # Gradienten für x, y, z
    title='3D Plot', filename='3d_plot.png',
    dpi=300, xlabel='X-Achse', ylabel='Y-Achse', zlabel='Z-Achse',
    colorbar_label='Farbwerte', path='Plots',
    quiver_stride=4, quiver_scale=0.05
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


        # Pfeile für die Gradienten (nur jeden zweiten Punkt)
        '''ax.quiver(
            x_values[mat_mask][::quiver_stride],
            y_values[mat_mask][::quiver_stride],
            z_values[mat_mask][::quiver_stride],
            u_grad[mat_mask][::quiver_stride],
            v_grad[mat_mask][::quiver_stride],
            w_grad[mat_mask][::quiver_stride],
            length=quiver_scale, arrow_length_ratio=0.1,
            color=matplotlib.colormaps.get_cmap(material_colormaps.get(mat, 'viridis'))(color_values[mat_mask][::quiver_stride]),
            alpha=0.7
        )'''

    # Achsenbeschriftung
    ax.set_xlabel(xlabel, color=kit_dark_blue, fontsize=12, labelpad=10)
    ax.set_ylabel(ylabel, color=kit_dark_blue, fontsize=12, labelpad=10)
    ax.set_zlabel(zlabel, color=kit_dark_blue, fontsize=12, labelpad=10)
    ax.set_title(title, color=kit_dark_blue, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5, linestyle='-')
    ax.tick_params(axis='x', colors=kit_dark_blue, direction='inout', length=6)
    ax.tick_params(axis='y', colors=kit_dark_blue, direction='inout', length=6)
    ax.tick_params(axis='z', colors=kit_dark_blue, direction='inout', length=6)

    # Farbskala
    sc = ax.scatter([], [], [], c=[], cmap=material_colormaps[np.unique(material_values)[0]],
                    vmin=color_values.min(), vmax=color_values.max())
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(colorbar_label, color=kit_dark_blue, fontsize=12)

    plt.show()

path_data = '..\\Archiv\\DataSets\\DataFiltered'#'..\\DataSets\\DataSimulated'
files = os.listdir(path_data)
materials = ['AL_2007_T4', 'S235JR']
df = []

for file in files:
    for mat in materials:
        if file.startswith(mat):
            data = pd.read_csv(f'{path_data}/{file}')
            # Gradienten berechnen (vor der Filterung)
            u = np.gradient(data['v_x'])
            v = np.gradient(-data['f_x_sim'])
            w = np.gradient(data['curr_x'])
            data['u_grad'] = u
            data['v_grad'] = v
            data['w_grad'] = w
            data['z_x'] = sign_hold(data['v_x'])
            data['material'] = mat
            df.append(data)
            break

data = pd.concat(df)
mask = (abs(data['v_x']) < 1)
data = data[mask]

mask = (data['z_x'] < 0)
data1 = data[mask]
mask = (data['z_x'] > 0)
data2 = data[mask]

for idx, data in enumerate([data1, data2]):
    if idx == 0:
        z = 'z_x < 0'
    else:
        z = 'z_x > 0'
    plot_3d_with_color(
        data['v_x'], -data['f_x_sim'], data['curr_x'], data['a_x'],
        material_values=data['material'], colorbar_label='$a$ in mm/s²',
        u_grad=data['u_grad'], v_grad=data['v_grad'], w_grad=data['w_grad'],
        xlabel='$v$ in m/s', ylabel='$F_{sim}$ in N', zlabel='$I$ in A',
        title=f'3D-Trajektorie für {z}',
        filename='3d_trajectory_by_material.png'
    )
