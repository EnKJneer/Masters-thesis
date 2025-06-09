import glob
import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsRegressor


def detect_discontinuity_knn(X, y, k=5, threshold=3.0):
    """
    Detektiert unstetige y-Werte bzgl. X durch lokale KNN-Regression.
    Gibt eine Liste von Indizes zurück, die als unstetig eingestuft wurden.

    Parameters:
        X: ndarray, shape (n_samples, n_features)
        y: ndarray, shape (n_samples,)
        k: Anzahl der Nachbarn
        threshold: Faktor über mittlerem Fehler (z.B. 3σ)
    """
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X, y)
    y_pred = knn.predict(X)
    residuals = np.abs(y - y_pred)

    # Definiere Unstetigkeit als "Ausreißer" im Fehler
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    discontinuous_idx = np.where(residuals > mean_res + threshold * std_res)[0]

    return discontinuous_idx

dataClass = hdata.Combined_PK_TrainVal
dataClass.window_size = 10
X_train, X_val, X_test, y_train, y_val, y_test = dataClass.load_data()

for x, y in zip(X_test, y_test):
    discont_idx = detect_discontinuity_knn(x.values, y.values)

    if len(discont_idx) == len(y.values):
        print("Alle y_test-Werte zeigen Unstetigkeit bzgl. x_test.")
    else:
        print(f"{len(discont_idx)} von {len(y.values)} Werten sind unstetig.")
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(y.index, y, label='curr_x', color='blue')
        plt.scatter(y.iloc[discont_idx].index, y.iloc[discont_idx], color='red', label='Discontinuity')
        plt.title('Verlauf von curr_x mit Unstetigkeiten')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.show()

