import pandas as pd

def find_large_changes(df):
    """
    Identifiziert den ersten großen Anstieg und den letzten großen Abfall in der Zeile `f_x_sim` eines DataFrames.
    Ein großer Anstieg oder Abfall wird als Änderung von mehr als 100 definiert.

    Parameters:
    df (pandas.DataFrame): Das DataFrame, das die Zeile `f_x_sim` enthält.

    Returns:
    tuple: Ein Tupel, das die Positionen (Indizes) des ersten großen Anstiegs und des letzten großen Rückgangs enthält.
    """
    # Angenommen df ist ein pandas DataFrame und 'f_x_sim' ist die Zeilenbeschriftung
    # Wir holen uns die Zeile als Series
    row = df.loc['f_x_sim']

    # Berechne die Differenzen zwischen aufeinanderfolgenden Werten
    diffs = row.diff().dropna()

    # Finde den ersten großen Anstieg (Differenz > 100)
    large_increase_indices = diffs[diffs > 100].index
    first_large_increase_index = large_increase_indices.min() if not large_increase_indices.empty else None

    # Finde den letzten großen Rückgang (Differenz < -100)
    large_decrease_indices = diffs[diffs < -100].index
    last_large_decrease_index = large_decrease_indices.max() if not large_decrease_indices.empty else None

    return first_large_increase_index, last_large_decrease_index

# Beispielanwendung:
# df = pd.DataFrame({'col1': [10, 120, 200, 150, 10], 'col2': [50, 200, 250, 100, 50]}, index=['f_x_sim', 'other_row'])
# first_increase, last_decrease = find_large_changes(df)
# print("Index des ersten großen Anstiegs:", first_increase)
# print("Index des letzten großen Rückgangs:", last_decrease)
if __name__ == "__main__":
    path_data = 'DataSets_CMX_Plate_Notch_Gear\Data'
    path_target = 'DataSets_CMX_Plate_Notch_Gear\Data'
