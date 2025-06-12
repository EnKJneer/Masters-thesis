import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path_data = 'DataFiltered'

files = os.listdir(path_data)
files = ['AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv',
         'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']

n = 25
for file in files:
    if '_3' in file:
        data = pd.read_csv(f'{path_data}/{file}')
        print(data.columns)
        print(data.shape)
        y_values_list = []
        y_labels_list = ['v_x']
        for ylabel_1 in y_labels_list:
            y_values_list.append(data[ylabel_1].iloc[:-n])

        epsilon = 1e-2
        v_x = data['v_x'].iloc[:-n]
        v_y = data['v_y'].iloc[:-n]

        v_x[np.abs(v_x) < epsilon] = 0
        v_y[np.abs(v_y) < epsilon] = 0

        #y_values_list.append(v_x**2)
        #y_labels_list.append('v_x**2')

        v = np.abs(v_x)/ (np.abs(v_x) + np.abs(v_y))

        mrr_x = v*data['materialremoved_sim'].iloc[:-n]
        #y_values_list.append(mrr_x)
        #y_labels_list.append('mrr_x')

        y_values_list.append(mrr_x*data['f_x_sim'].iloc[:-n])
        y_labels_list.append('f_x_sim * mrr_x')

        y_values_list.append(mrr_x*np.abs(data['f_x_sim'].iloc[:-n]) * np.sign(v_x))
        y_labels_list.append('f_x_sim * mrr_x * sign(v_x)')

        for y_label_1, y_values_1 in zip(y_labels_list, y_values_list):
            xlabel = 'time'
            ylabel_2 = 'curr_x'

            x_values = data.index[:-n]

            y_values_2 = -data[ylabel_2].iloc[:-n]

            # Plot erstellen
            fig, ax1 = plt.subplots()

            # Erste y-Achse
            color = 'tab:blue'
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(y_label_1, color=color)
            ax1.plot(x_values, y_values_1, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            # Zweite y-Achse
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('-'+ylabel_2, color=color)
            ax2.plot(x_values, y_values_2, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            name = file.replace('.csv', '')
            # Titel setzen
            plt.title(f'{name} von 0 bis ende-{n}')

            # Plot anzeigen
            plt.show()