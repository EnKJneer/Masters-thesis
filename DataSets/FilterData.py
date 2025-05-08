import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.ao.nn.quantized.functional import threshold

HEADER = ["pos_x", "pos_y", "pos_z", "v_sp", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "a_sp", "curr_x", "curr_y", "curr_z", "curr_sp", "f_x_sim", "f_y_sim", "f_z_sim", "f_sp_sim", "materialremoved_sim"]

target_path = 'DataFiltered'
folder_path = 'RawData'
file_names = glob.glob(os.path.join(folder_path, "*.csv"))

for file_name in file_names:
    # Datei einlesen
    df = pd.read_csv(file_name)

    # Median berechnen
    median_x = df['pos_x'].median()
    median_y = df['pos_y'].median()

    # Entfernung vom Median berechnen
    df['distance_from_median'] = np.sqrt((df['pos_x'] - median_x) ** 2 + (df['pos_y'] - median_y) ** 2)
    df['v'] =  np.sqrt((df['v_x']) ** 2 + (df['v_y']) ** 2)
    # MAD berechnen
    mad_x = np.median(np.abs(df['pos_x'] - median_x))
    mad_y = np.median(np.abs(df['pos_y'] - median_y))

    # Modifizierter Z-Score berechnen
    df['mod_z_score_x'] = 0.6745 * (df['pos_x'] - median_x) / mad_x
    df['mod_z_score_y'] = 0.6745 * (df['pos_y'] - median_y) / mad_y

    threshold = 3
    # Ausreißer entfernen (z.B. alle Punkte mit einem modifizierten Z-Score > 3.5)
    df_cleaned = df[(df['mod_z_score_x'].abs() < threshold) & (df['mod_z_score_y'].abs() < threshold)]

    # Plot erstellen
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df_cleaned['pos_x'], df_cleaned['pos_y'], c=df_cleaned['v'], cmap='viridis', label='Positionen')

    # Median im Plot darstellen
    plt.scatter(median_x, median_y, color='red', zorder=5, label='Median')

    # Farbskala hinzufügen
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Geschwindgkeit')

    # Titel und Legende hinzufügen
    plt.title(file_name.split('\\')[-1])
    plt.xlabel('pos_x')
    plt.ylabel('pos_y')
    plt.legend()

    # Plot anzeigen
    plt.show()

    df_cleaned[HEADER].to_csv(os.path.join(target_path, file_name.split('\\')[-1]), index=False)

