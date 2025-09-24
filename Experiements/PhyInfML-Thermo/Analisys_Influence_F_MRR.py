import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import Helper.handling_data as hdata

def load_csv_files(csv_paths):
    """
    Lädt mehrere CSV-Dateien in ein DataFrame.

    Args:
        csv_paths (list): Liste der Pfade zu den CSV-Dateien
    """
    # Alle CSV-Dateien in einen DataFrame laden
    dfs = []
    for path in csv_paths:
        file_path = os.path.join(hdata.DataClass_ST_Plate_Notch.folder, path)
        df = pd.read_csv(file_path)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df

def plot_df(combined_df, column1, column2):
    """
    Erstellt einen Scatterplot
    zweier Spalten und berechnet deren Korrelation.

    Args:
        column1 (str): Name der ersten Spalte für den Scatterplot
        column2 (str): Name der zweiten Spalte für den Scatterplot
    """
    # Scatterplot erstellen
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=combined_df, x=column1, y=column2)
    plt.title(f'Scatterplot: {column1} vs {column2}')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.grid(True)
    plt.show()

    # Korrelation berechnen
    correlation = combined_df[[column1, column2]].corr().iloc[0, 1]
    print(f"\nKorrelation zwischen '{column1}' und '{column2}': {correlation:.4f}")

# Beispielaufruf
if __name__ == "__main__":
    # Pfade zu den CSV-Dateien
    csv_paths = hdata.DataClass_ST_Plate_Notch.training_data_paths

    df = load_csv_files(csv_paths)
    df['term'] = df['f_x_sim'] * df['materialremoved_sim']

    # Spaltennamen für den Scatterplot
    column1 = "curr_x"  # Ersetze mit deinem Spaltennamen

    for col in ['f_x_sim', 'materialremoved_sim', 'term']:
        # Funktion aufrufen
        plot_df(df, column1, col)
