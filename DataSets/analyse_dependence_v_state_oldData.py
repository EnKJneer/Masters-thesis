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
    'CMX_Alu_Tr_Air_2_alldata_allcurrent.csv': ('lightgreen', 'AL Plate Depth 1'),
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
    'CMX_Alu_Val_Air_2_alldata_allcurrent.csv': ('red', 'S Plate 1'),
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

path_data = 'OldDataSets'
files = [
    #'CMX_Alu_Tr_Air_2_alldata_allcurrent.pkl',
    'CMX_Alu_Val_Air_2_alldata_allcurrent.pkl',
    #'I40_Alu_Tr_Air_2_alldata_allcurrent.pkl',
    'I40_Alu_Val_Air_2_alldata_allcurrent.pkl',
    #'CMX_St_Tr_Air_2_alldata_allcurrent.pkl',
    #'CMX_St_Val_Air_2_alldata_allcurrent.pkl',
    #'I40_St_Tr_Air_2_alldata_allcurrent.pkl',
    #'I40_St_Val_Air_2_alldata_allcurrent.pkl',
]

n = 25
axes = ['x']

def sign_hold(v, eps=1e-1):
    z = np.zeros(len(v))
    h = deque([1, 1, 1, 1, 1], maxlen=5)
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])
        if i >= 4:
            z[i] = np.sign(sum(h))
    return z

fig, axs = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
fig.suptitle('CMX and I40 Data Alu_Val_Air_2_alldata_allcurrent')

for axis in axes:
    for file in files:
        data = pd.read_pickle(f'{path_data}/{file}')
        header = ['pos_x', 'pos_y', 'pos_z', 'pos_sp', 'curr_x', 'curr_y', 'curr_z', 'curr_sp']
        data = pd.DataFrame(data.T)
        data.columns = header
        epsilon = 1e-1
        data[f'v_{axis}'] = data[f'pos_{axis}'].diff()
        data[f'a_{axis}'] = data[f'v_{axis}'].diff()
        a_axis = data[f'a_{axis}'].iloc[:-n].copy()
        v_axis = data[f'v_{axis}'].iloc[:-n].copy()
        curr_axis = -data[f'curr_{axis}'].iloc[:-n].copy()
        time = data.index[:-n].copy()
        mask = (v_axis.abs() <= 1000)
        v_axis = v_axis[mask].reset_index(drop=True)
        a_axis = a_axis[mask].reset_index(drop=True)
        curr_axis = curr_axis[mask].reset_index(drop=True)
        z = sign_hold(v_axis)
        time = time[mask]
        d = pd.Series(v_axis)
        window = 100
        v_axis = np.array(d.rolling(window).mean())
        d = pd.Series(curr_axis)
        curr_axis = np.array(d.rolling(window).mean().fillna(0))
        color, label = file_colors.get(file, ('black', file))

        sort_idx = np.argsort(v_axis)
        y_datas = [("curr_axis", curr_axis)]
        color_value = ("time", time)

        for y_data in y_datas:
            if file.startswith('CMX'):
                ax = axs[0]
                ax.set_title('CMX Data')
            elif file.startswith('I40'):
                ax = axs[1]
                ax.set_title('I40 Data')

            ax.scatter(v_axis, y_data[1], c=color_value[1], s=2, alpha=0.5, label=label)
            ax.set_xlabel(f'v_{axis}')
            ax.set_ylabel(y_data[0])

            dx = []
            dy = []
            for i in range(len(v_axis) - 1):
                dx.append(v_axis[i + 1] - v_axis[i])
                dy.append(y_data[1][i + 1] - y_data[1][i])
            ax.quiver(v_axis[:-1], y_data[1][:-1], dx, dy, angles='xy', scale_units='xy', scale=1, width=0.005,
                      color='gray', alpha=0.5)

            xlimit = 2
            ylimit = 3
            #ax.set_xlim(max(-xlimit, min(v_axis)*1.1), min(xlimit, max(v_axis)*1.1))
            #ax.set_ylim(max(-ylimit, min(y_data[1])*1.1), min(ylimit, max(y_data[1])*1.1))

            norm = mcolors.Normalize(vmin=np.min(color_value[1]), vmax=np.max(color_value[1]))
            sm = plt.cm.ScalarMappable(norm=norm)
            sm.set_array([])

            plt.colorbar(sm, ax=ax, label=color_value[0])

plt.tight_layout()
plt.show()
