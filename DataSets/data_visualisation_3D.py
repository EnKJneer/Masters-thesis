import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module

mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer

"""
This script processes and visualizes anomaly data compared to normal reference data.
It performs the following tasks:

1. Loads anomaly and corresponding normal data from CSV files.
2. Calculates velocity and acceleration from position data.
3. Trims the data to the minimum length of the two datasets.
4. Computes the difference between the anomaly and normal datasets.
5. Loads corresponding drilling positions for visualization.
6. Plots 3D scatter plots of positions with current values (curr_x, curr_y, curr_sp).
7. Plots time series data for current, velocity, and acceleration along x, y, and sp axes.

Dependencies:
- numpy
- pandas
- os
- matplotlib
- mpl_toolkits.mplot3d

Usage:
- Ensure the data files are organized in the specified directory structure.
- Run the script to generate and display the plots.

Note:
- The script assumes a specific naming convention for the data files.
- The drilling positions are visualized as green lines in the 3D plots.
"""

plot_tool_wear = False
plot_blowhole = False
plot_positions = True
plot_all = True
axes = ['curr_x', 'curr_y', 'curr_sp']

# load data
dataSets = ''
folder_path =  dataSets+'DataFiltered'
files = os.listdir(folder_path)
files = ['AL_2007_T4_Plate_Normal_3.csv']
for filename in files:

    if '_3' in filename:
        data = pd.read_csv(os.path.join(folder_path, filename))

        # calculate velocity
        col_pos = ['pos_x', 'pos_y', 'pos_z']
        col_vel = ['vel_x', 'vel_y', 'vel_sp']
        col_acc = ['acc_x', 'acc_y', 'acc_sp'] #
        for idx, col in enumerate(col_pos):
            vel = col_vel[idx]
            acc = col_acc[idx]
            data[vel] = data[col].diff() / 50
            data[acc] = data[vel].diff() / 50

        """    # load corresponding drilling positions
        if 'Blowhole' in filename:
            drilling_filename= filename.replace('Blowhole', 'Drilling_Position')
           # There are no drilling data for gear
            if 'Gear' in drilling_filename:
                drilling_positions = pd.DataFrame()
            else:
                # load drilling position
                drilling_positions = pd.read_csv(dataSets + '/DataDrilling/' + drilling_filename)
    
            # Define the tolerance
            tolerance = 0.1
    
            # Function to check if values are within tolerance
            def is_within_tolerance(value1, value2, tolerance):
                return abs(value1 - value2) <= tolerance
    
            # List to store the matching time values
            matching_times = []
    
            # Iterate through the drilling_positions
            for idx, row in drilling_positions.iterrows():
                pos_x_drill = row['pos_x']
                pos_y_drill = row['pos_y']
    
                # Find the corresponding time values in data_anomalie
                for idx_anomalie, row_anomalie in data.iterrows():
                    if is_within_tolerance(row_anomalie['pos_x'], pos_x_drill, tolerance) and is_within_tolerance(
                            row_anomalie['pos_y'], pos_y_drill, tolerance):
                        matching_times.append(row_anomalie['time'])
        """
        data['time'] = data.index / 50
        if plot_positions:
            axes = ['curr_x', 'curr_y'] # , 'curr_y', 'curr_sp'
            # plot position
            for axis in axes:
                # Plot the position with current X
                plt.figure(figsize=(12, 8))
                ax = plt.axes(projection='3d')  # Create a 3D axis

                label = axis
                max_value = 2  # -3 for curr_y # 2 for curr_x
                min_value = -2  # -7 for curr_y # -2 for curr_x
                color_values = np.clip(data[label], min_value, max_value)

                # Scatter plot with three axes
                sc = ax.scatter(
                    data['pos_x'],
                    data['pos_y'],
                    data[axis],
                    c=color_values,
                    cmap='viridis',
                    label=axis,
                    s=0.1
                )

                # Hinzufügen einer Farbskala
                plt.colorbar(sc, label=axis)

                # Set axis labels
                ax.set_xlabel('Pos_X')
                ax.set_ylabel('Pos_Y')
                ax.set_zlabel(label)

                # Title and legend
                plt.title(filename.replace('.csv', ': ') + 'Positions and ' + axis)
                plt.legend()
                plt.show()

        if plot_all:
            # Axes and metrics
            axes = ['x', 'y', 'sp']
            metrics = ['curr', 'pos', 'vel', 'acc']

            # plot time series
            rows = len(axes)
            cols = len(metrics)
            # Create a figure and a grid of subplots
            fig, axs = plt.subplots(rows, cols, figsize=(18, 12))

            # Loop over the axes and metrics
            for i, axis in enumerate(axes):
                for j, metric in enumerate(metrics):
                    if metric == 'pos' and axis == 'sp':
                        axis = 'z'
                    if metric == 'vel' and axis == 'z':
                        axis = 'sp'
                    key_anomalie = f'{metric}_{axis}'
                    key_normal = f'{metric}_{axis}'

                    axs[i, j].scatter(data['time'], data[key_anomalie], color='blue',
                                      label=metric+axis, s=0.1)
                    axs[i, j].set_xlabel('time')
                    axs[i, j].set_ylabel(key_anomalie)
                    axs[i, j].set_title(f'{metric.capitalize()} ({axis}-axis)')
                    axs[i, j].legend()

                    """                # Add green vertical lines for matching times if 'Blowhole' is in filename and matching_times is not empty
                    if 'Blowhole' in filename:
                        if len(drilling_positions) > 0 and metric == 'pos' and not(axis == 'z'):
                            for pos in drilling_positions[key_anomalie]:
                                if data[key_anomalie].min() < pos < data[key_normal].max():
                                    axs[i, j].axhline(y=pos, color='green', linestyle='--', linewidth=1)
                        '''
                        if len(matching_times) > 0:
                            for time in matching_times:
                                axs[i, j].axvline(x=time, color='green', linestyle='--', linewidth=1)
                        '''
                    """
            # Title for the entire figure
            fig.suptitle(filename.replace('.csv', ''))

            # Display the plot
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()