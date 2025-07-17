import os
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def sign_hold(v, eps=1e-1):
    z = np.zeros(len(v))
    h = deque([1, 1, 1, 1, 1], maxlen=5)
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])
        if i >= 4:
            z[i] = np.sign(sum(h))
    return z

def plot_2d_with_color(x_values, y_values, color_values, filename, label='|v_x + v_y|', title='2D Plot von pos_x und pos_y mit Farbe', dpi=300, xlabel='pos_x', ylabel='pos_y'):
    plt.figure(figsize=(10, 6), dpi=dpi)
    normalized_color_values = (color_values - np.min(color_values)) / (np.max(color_values) - np.min(color_values))
    sc = plt.scatter(x_values, y_values, c=color_values, cmap='viridis', s=1)
    plt.colorbar(sc, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_time_series(data, title, label='v_x', ylabel='curr_x', dpi=300):
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax1.plot(data.index, data[label], label=label, color='tab:green')
    ax1.set_xlabel('Index')
    ax1.set_ylabel(label)
    ax1.set_title(title)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(data.index, data[ylabel], label=ylabel, color='tab:red')
    ax2.set_ylabel(ylabel)
    ax2.legend(loc='upper right')

    # Find the first index where materialremoved_sim is greater than 0
    first_index = data[data['materialremoved_sim'] > 0].index.min()
    if pd.notna(first_index):
        ax1.axvline(x=first_index, color='black', linestyle='--', label='First material removed')
    plt.show()


path_data = 'Data'
files = ['S235JR_Plate_Normal_1.csv']
files = ['AL_2007_T4_Gear_Depth_1.csv', 'AL_2007_T4_Plate_Depth_1.csv']
for file in files:
    data = pd.read_csv(f'{path_data}/{file}')
    n = data[data['materialremoved_sim'] > 0].index.min() *0.9
    data = data.iloc[:int(n), :]

    xlabel = 'pos_x'
    ylabel = 'pos_y'
    data['v'] = np.sqrt(data['v_x']**2 + data['v_y']**2)
    data['v'] = np.clip(data['v'], 5.83, 6)

    label = 'materialremoved_sim'
    x_values = data[xlabel]
    y_values = data[ylabel]
    color_values = data[label]
    #max_value = 2
    #min_value = -2
    #color_values = np.clip(color_values, min_value, max_value)

    name = file.replace('.csv', '')
    #plot_2d_with_color(x_values, y_values, color_values, f'Plots/{name}_{xlabel}_{label}', label=label, title=file, dpi=600, xlabel=xlabel, ylabel=ylabel)
    #plot_time_series(data, name, label='materialremoved_sim', dpi=300)
    #plot_time_series(data, name, label='v_x', dpi=300)
    #plot_time_series(data, name, label='f_x_sim', dpi=300)
    #plot_time_series(data, name, label='v_y', dpi=300, ylabel='curr_y')
    #plot_time_series(data, name, label='v_z', dpi=300, ylabel='curr_z')
    #plot_time_series(data, name, label='f_sp_sim', dpi=300, ylabel='curr_sp')

    #plot_time_series(data, name, label='pos_x', dpi=300, ylabel='curr_y')
    plot_time_series(data, name, label='pos_y', dpi=300, ylabel='curr_x')

