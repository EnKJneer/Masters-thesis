import pandas as pd
import matplotlib.pyplot as plt

# Lade die CSV-Datei
file = 'AL_2007_T4_Plate_SF.csv'
df = pd.read_csv(file)

# Definiere die zu plottenden Spalten
columns_to_plot = ['curr_x', 'v_x', 'f_x_sim', 'materialremoved_sim']

# Erstelle einen Plot mit mehreren Subplots
fig, axs = plt.subplots(len(columns_to_plot), 1, figsize=(10, 12), sharex=True)

# Setze den Titel des gesamten Plots auf den Dateinamen
fig.suptitle(file, fontsize=16)

# Plotte jede Spalte in einem Subplot
for i, column in enumerate(columns_to_plot):
    axs[i].plot(df.index, df[column], label=column)
    axs[i].set_ylabel(column)
    axs[i].set_title(f'Zeitlicher Verlauf von {column}')
    axs[i].grid(True)

# Setze die x-Achsenbeschriftung nur für den letzten Subplot
axs[-1].set_xlabel('Zeit')

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Passe das Layout an, um Platz für den suptitle zu machen
plt.show()
