import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pysindy as ps
import Models.model_base as mb
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn
import Models.model_physical as mphys
import Models.model_random_forest as mrf
import Models.model_mixture_of_experts as mmix
import Models.JAX_Version.model_neural_net as jmnn
import Models.JAX_Version.model_physical as jmphys
from pysindy import differentiation


class MachineAndProcessSINDy:
    def __init__(self, feature_names=None, differentiation_method='finite_difference', optimizer=None,
                 feature_library=None, dt=0.02):
        """
        feature_names: Liste der Feature-Namen in X (z.B. ['x1', 'x2', ...])
        differentiation_method: Methode für Ableitungen, z.B. 'finite_difference' oder 'smoothed_finite_difference'
        optimizer: z.B. ps.STLSQ(alpha=0.1)
        feature_library: z.B. ps.PolynomialLibrary(degree=3)
        dt: Zeitschritt für die Simulation
        """
        self.feature_names = feature_names
        self.differentiation_method = differentiation_method
        self.dt = dt

        # Optimizer und Feature Library Defaults
        self.optimizer = optimizer if optimizer is not None else ps.STLSQ(alpha=0.00001, threshold=0.005)
        self.feature_library = feature_library if feature_library is not None else ps.PolynomialLibrary(degree=3)

        self.model = None
        self.is_fitted = False

    def _prepare_data(self, X_list, y_list):
        """
        Nimmt Listen von DataFrames X und y, bereitet numpy-Arrays vor,
        berechnet numerisch dy/dt.
        Rückgabe:
            X_all: (N_total, n_features)
            y_all: (N_total, 1)
            dy_all: (N_total, 1) numerische Ableitung von y
        """
        X_concat = []
        y_concat = []
        dy_concat = []

        for X_df, y_df in zip(X_list, y_list):
            # 1) Gemeinsame Indizes
            common_idx = X_df.index.intersection(y_df.index)
            X_clean = X_df.loc[common_idx].dropna()
            y_clean = y_df.loc[common_idx].dropna()

            # Falls ungleich viele Punkte nach dropna: Angleichen
            n = min(len(X_clean), len(y_clean))
            X_clean = X_clean.iloc[:n]
            y_clean = y_clean.iloc[:n]

            # 2) Werte als numpy
            X_np = X_clean.values
            y_np = y_clean.values.reshape(-1, 1)  # Spalte

            # 3) Zeitvektor (gleichmäßiger Zeitschritt)
            t = np.arange(len(y_np)) * self.dt

            # 4) Numerische Ableitung von y
            differentiator = differentiation.FiniteDifference(order=2)
            dy_dt = differentiator._differentiate(y_np, t)

            X_concat.append(X_np)
            y_concat.append(y_np)
            dy_concat.append(dy_dt)

        X_all = np.vstack(X_concat)
        y_all = np.vstack(y_concat)
        dy_all = np.vstack(dy_concat)

        return X_all, y_all, dy_all

    def train_model(self, X_list, y_list):
        """
        Trainiere PySINDy-Modell für die Dynamik dy/dt = f(x, y)
        """
        # 1) Daten vorbereiten
        X_all, y_all, dy_all = self._prepare_data(X_list, y_list)

        # 2) Kombiniere x und y als Input-Features für SINDy
        XY = np.hstack([X_all, y_all])  # shape (N, n_features + 1)

        # 3) Zeitvektor für alle Datenpunkte
        t = np.arange(len(XY)) * self.dt

        # 4) Modell initialisieren und fitten
        self.model = ps.SINDy(
            optimizer=self.optimizer,
            feature_library=self.feature_library,
            differentiation_method=ps.FiniteDifference(order=2)
        )

        # SINDy fit mit Zeitvektor und Zielvariable dy/dt
        self.model.fit(XY, t=t, x_dot=dy_all)
        self.is_fitted = True

    def predict_derivative(self, X, y):
        """
        Gibt dy/dt vorhergesagt für Input X und y zurück
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")

        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        y_np = y.values.reshape(-1, 1) if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame) else np.array(
            y).reshape(-1, 1)
        XY = np.hstack([X_np, y_np])

        # Verwende predict um dy/dt zu erhalten (direkter Ansatz)
        return self.model.predict(XY)

    def simulate_y(self, X: pd.DataFrame, y0: float, dt: float = None):
        """
        Simuliert y über die Zeit basierend auf den Inputs X und Startwert y0
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first!")

        if dt is None:
            dt = self.dt

        n_steps = len(X)
        y_sim = np.zeros(n_steps)
        y_sim[0] = y0

        for i in range(n_steps - 1):
            # Aktueller Zustand: [X[i], y[i]]
            current_state = np.hstack([X.iloc[i].values, y_sim[i]])

            # Vorhersage der Ableitung dy/dt mit predict
            dydt = self.model.predict(current_state.reshape(1, -1))

            # Euler-Integration für nächsten Zeitschritt
            y_sim[i + 1] = y_sim[i] + dydt[0, -1] * dt  # dydt[-1] ist dy/dt (letzter Wert)

        return y_sim

    def get_model_equations(self):
        """
        Gibt die erkannten Gleichungen als String zurück
        """
        if not self.is_fitted:
            return "Model is not fitted yet."
        return self.model.print()


if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 1
    window_size = 10
    past_values = 0
    future_values = 0

    dataclass2 = hdata.Combined_Plate
    dataclass2.target_channels = ['curr_x']
    dataClasses = [dataclass2]

    for dataClass in dataClasses:
        dataClass.remove_bias = False
        dataClass.window_size = window_size
        dataClass.past_values = past_values
        dataClass.future_values = future_values
        dataClass.keep_separate = True

    # Daten laden
    X_train, X_val, X_test, y_train, y_val, y_test = dataClass.load_data()

    # Modell instanziieren
    model_sindy = MachineAndProcessSINDy(feature_names=X_train[0].columns.tolist())

    # Trainieren
    model_sindy.train_model(X_train, y_train)

    # Vorhersage von dy/dt auf neuen Daten (z.B. X_test, y_test)
    dy_pred = model_sindy.predict_derivative(X_test[0], y_test[0])

    # Simulation aus Startwert y0 und Inputs
    #y_sim = model_sindy.simulate_y(X_test[0], y0=float(y_test[0].iloc[0]), dt=0.02)

    # Gleichungen anzeigen
    print(model_sindy.get_model_equations())

    # Optional: Vergleich der Ergebnisse
    #print(f"Simulationsergebnisse: {y_sim[:5]}")  # Erste 5 Werte
    #print(f"Tatsächliche Werte: {y_test[0].iloc[:5].values}")  # Erste 5 Werte