import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path_data = 'DataFiltered'
files = [
    #'AL_2007_T4_Plate_Normal_3.csv', 'AL_2007_T4_Gear_Normal_3.csv',
    #'AL_2007_T4_Plate_SF_3.csv', 'AL_2007_T4_Gear_SF_3.csv',
    'AL_2007_T4_Plate_Depth_1.csv', 'AL_2007_T4_Gear_Depth_1.csv',
    'AL_2007_T4_Plate_Depth_2.csv', 'AL_2007_T4_Gear_Depth_2.csv',
    'AL_2007_T4_Plate_Depth_3.csv', 'AL_2007_T4_Gear_Depth_3.csv',
    #'S235JR_Gear_Normal_3.csv', 'S235JR_Plate_Normal_3.csv',
    #'S235JR_Plate_SF_3.csv', 'S235JR_Gear_SF_3.csv',
    'S235JR_Plate_Depth_1.csv','S235JR_Gear_Depth_1.csv',
    'S235JR_Plate_Depth_2.csv','S235JR_Gear_Depth_2.csv',
    'S235JR_Plate_Depth_3.csv','S235JR_Gear_Depth_3.csv'
]

n = 25
plt.figure(figsize=(10, 6))

# Define colors and labels for each file
file_colors = {
    'AL_2007_T4_Plate_Normal_3.csv': ('blue', 'AL Plate'),
    'AL_2007_T4_Gear_Normal_3.csv': ('green', 'AL Gear'),
    'AL_2007_T4_Plate_SF_3.csv': ('cyan', 'AL Plate SF'),
    'AL_2007_T4_Plate_Depth_1.csv': ('lightgreen', 'AL Plate Depth 1'),
    'AL_2007_T4_Plate_Depth_2.csv': ('darkgreen', 'AL Plate Depth 2'),
    'AL_2007_T4_Plate_Depth_3.csv': ('olivedrab', 'AL Plate Depth 3'),
    'AL_2007_T4_Gear_SF_3.csv': ('teal', 'AL Gear SF'),
    'AL_2007_T4_Gear_Depth_1.csv': ('mediumblue', 'AL Gear Depth 1'),
    'AL_2007_T4_Gear_Depth_2.csv': ('navy', 'AL Gear Depth 2'),
    'AL_2007_T4_Gear_Depth_3.csv': ('darkblue', 'AL Gear Depth 3'),
    'S235JR_Gear_Normal_3.csv': ('orange', 'S Gear'),
    'S235JR_Plate_Normal_3.csv': ('red', 'S Plate'),
    'S235JR_Plate_SF_3.csv': ('pink', 'S Plate SF'),
    'S235JR_Plate_Depth_1.csv': ('sandybrown', 'S Plate Depth 1'),
    'S235JR_Plate_Depth_2.csv': ('brown', 'S Plate Depth 2'),
    'S235JR_Plate_Depth_3.csv': ('saddlebrown', 'S Plate Depth 3'),
    'S235JR_Gear_SF_3.csv': ('purple', 'S Gear SF'),
    'S235JR_Gear_Depth_1.csv': ('lightgray', 'S Gear Depth 1'),
    'S235JR_Gear_Depth_2.csv': ('gray', 'S Gear Depth 2'),
    'S235JR_Gear_Depth_3.csv': ('darkgray', 'S Gear Depth 3'),
}

'''
for file in files:
    data = pd.read_csv(f'{path_data}/{file}')
    y_values_list = []
    y_labels_list = ['f_x_sim']
    for ylabel_1 in y_labels_list:
        y_values_list.append(data[ylabel_1].iloc[:-n])

    epsilon = 1e-2
    v_x = data['v_x'].iloc[:-n]
    v_y = data['v_y'].iloc[:-n]
    v_x[np.abs(v_x) < epsilon] = 0
    v_y[np.abs(v_y) < epsilon] = 0
    v = np.abs(v_x) / (np.abs(v_x) + np.abs(v_y))

    for y_label_1, y_values_1 in zip(y_labels_list, y_values_list):
        y_label_2 = 'curr_x'
        x_values = data.index[:-n]
        y_values_2 = -data[y_label_2].iloc[:-n]

        # Get color and label for the current file
        color, label = file_colors[file]

        # Plot data for the current file
        plt.scatter(y_values_1, y_values_2, c=color, s=2, label=label)

# Set labels and title
plt.xlabel('f_x_sim')
plt.ylabel('curr_x')
plt.title('Combined Plot for All Files')

# Add legend
plt.legend()
# Show plot
plt.show()

for file in files:
    data = pd.read_csv(f'{path_data}/{file}')
    y_values_list = []
    y_labels_list = ['v_x']
    for ylabel_1 in y_labels_list:
        y_values_list.append(data[ylabel_1].iloc[:-n])

    epsilon = 1e-2
    v_x = data['v_x'].iloc[:-n]
    v_y = data['v_y'].iloc[:-n]
    v_x[np.abs(v_x) < epsilon] = 0
    v_y[np.abs(v_y) < epsilon] = 0
    v = np.abs(v_x) / (np.abs(v_x) + np.abs(v_y))

    for y_label_1, y_values_1 in zip(y_labels_list, y_values_list):
        y_label_2 = 'curr_x'
        x_values = data.index[:-n]
        y_values_2 = -data[y_label_2].iloc[:-n]

        # Get color and label for the current file
        color, label = file_colors[file]

        # Plot data for the current file
        plt.scatter(y_values_1, y_values_2, c=color, s=2, label=label)

# Set labels and title
plt.xlabel('v_x')
plt.ylabel('curr_x')
plt.title('Combined Plot for All Files')

# Add legend
plt.legend()
# Show plot
plt.show()
'''
# Create a figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(21, 6), dpi=300)

for file in files:
    data = pd.read_csv(f'{path_data}/{file}')

    epsilon = 1e-2
    v_x = data['v_x'].iloc[:-n]
    a_x = data['a_x'].iloc[:-n]
    v_x[np.abs(v_x) < epsilon] = 0
    mrr = data['materialremoved_sim'].iloc[:-n]
    # Filter out values where |a_x| > 1
    mask_a_x = (np.abs(a_x) <= 1000)

    v_x = v_x[mask_a_x]
    f_x_sim = data['f_y_sim'].iloc[:-n][mask_a_x]
    curr_x = -data['curr_x'].iloc[:-n][mask_a_x]


    # Get color and label for the current file
    color, label = file_colors[file]

    # Plot data for v_x > 0
    mask_gt_zero = v_x > 0
    axs[0].scatter(f_x_sim[mask_gt_zero], curr_x[mask_gt_zero], c=color, s=2, label=label)
    axs[0].set_xlabel('f_y_sim')
    axs[0].set_ylabel('curr_x')
    axs[0].set_title('Plot for v_x > 0')
    axs[0].legend()

    # Plot data for v_x < 0
    mask_lt_zero = v_x < 0
    axs[1].scatter(f_x_sim[mask_lt_zero], curr_x[mask_lt_zero], color=color, s=2, label=label)
    axs[1].set_xlabel('f_y_sim')
    axs[1].set_ylabel('curr_x')
    axs[1].set_title('Plot for v_x < 0')
    axs[1].legend()

    # Plot data for v_x == 0
    mask_eq_zero = v_x == 0
    axs[2].scatter(f_x_sim[mask_eq_zero], curr_x[mask_eq_zero], c=color, s=2, label=label)
    axs[2].set_xlabel('f_x_sim')
    axs[2].set_ylabel('curr_x')
    axs[2].set_title('Plot for v_x == 0')
    axs[2].legend()

plt.tight_layout()
plt.show()