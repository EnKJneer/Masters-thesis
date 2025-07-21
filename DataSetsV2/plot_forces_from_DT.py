# Hauptverarbeitung
import os

import matplotlib.pyplot as plt
import pandas as pd

path_data = 'RawData'

files = os.listdir(path_data)
files = ['AL_2007_T4_Plate_Normal.csv']
for file in files:
    if not file.endswith('.csv'):
        continue
    print(f'Processing {file}')
    df = pd.read_csv(os.path.join(path_data, file))
    df.iloc[:,1:] = df.iloc[:,1:].rolling(window=10, min_periods=1).mean()
    e = int(len(df.index))
    for i in range(0, 2):
        print(f'Plot {i}')
        n = 20000
        plt.plot(df['Time'].iloc[:e:n], df[f'DT9836(00)_{i}'].iloc[:e:n], label = f'DT {i}')
        plt.legend()
        plt.show()

"""
DT 1: Kraft x
DT 2: Kraft y
DT 3: Kraft z
DT 4: Beschleunigung x (vermutlich)
DT 5: Beschleunigung y (vermutlich)
DT 6: Beschleunigung z (vermutlich)
"""