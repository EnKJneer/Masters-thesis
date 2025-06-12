import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Pfad zu den Dateien
path_data = 'DataFiltered'

# Liste der Dateien
files = ['AL_2007_T4_Plate_Normal_3.csv', 'S235JR_Plate_Normal_3.csv']
n = 25


# Iteriere über die Dateien
for file in files:
    data = pd.read_csv(os.path.join(path_data, file))
    print(f"Columns in {file}: {data.columns}")
    print(f"Shape of data in {file}: {data.shape}")

    data = data.iloc[:-n]

    v_x = data['v_x']
    v_y = data['v_y']
    v_z = data['v_z']
    data['mrr_x'] = data['materialremoved_sim'] * (np.abs(v_x) / (np.abs(v_x) + np.abs(v_y) + 1e-10))
    data['mrr_y'] = data['materialremoved_sim'] * (np.abs(v_y) / (np.abs(v_x) + np.abs(v_y) + 1e-10))
    data['mrr_z'] = data['materialremoved_sim'] * (np.abs(v_z) / (np.abs(v_x) + np.abs(v_y) + 1e-10))
    axes = ['x', 'y']

    for axis in axes:
        eps = 0.001
        data_filtered = data.where(np.abs(data[f'v_{axis}']) < eps)
        data_filtered = data_filtered.dropna()
        data_filtered = data_filtered.set_axis(range(len(data_filtered)))
        y = data_filtered[f'curr_{axis}']
        x = data_filtered[['v_x', 'v_y', 'f_x_sim', 'f_y_sim', 'materialremoved_sim']]

        # Polynomiale Regression
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(x)

        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        y_pred_poly = poly_model.predict(X_poly)

        # Random Forest Regression
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(x, y)
        y_pred_rf = rf_model.predict(x)

        # Feature Importance für Random Forest
        feature_importance = rf_model.feature_importances_

        # Berechnung des mittleren quadratischen Fehlers (MSE)
        poly_mse = np.mean((y_pred_poly - y) ** 2)
        rf_mse = np.mean((y_pred_rf - y) ** 2)

        print(f"{file} Mean Squared Error (Polynomiale Regression) für {axis}-Achse: {poly_mse}")
        print(f"{file} Mean Squared Error (Random Forest) für {axis}-Achse: {rf_mse}")
        print(f"{file} Feature Importance (Random Forest) für {axis}-Achse: {feature_importance}")

        # Plotting des zeitlichen Verlaufs
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(y.index, y, label='Tatsächliche Werte', color='blue')
        plt.plot(y.index, y_pred_poly, label='Vorhergesagte Werte', linestyle='--', color='red')
        plt.xlabel('Zeitindex')
        plt.ylabel('Werte')
        plt.title(f'{file}: Polynomiale Regression für {axis}-Achse')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(y.index, y, label='Tatsächliche Werte', color='blue')
        plt.plot(y.index, y_pred_rf, label='Vorhergesagte Werte', linestyle='--', color='green')
        plt.xlabel('Zeitindex')
        plt.ylabel('Werte')
        plt.title(f'Random Forest Regression für {axis}-Achse')
        plt.legend()

        plt.tight_layout()
        plt.show()

