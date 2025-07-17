import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import optuna
from sklearn.ensemble import RandomForestRegressor as mrf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_base as mb


class MachineAndProcessARMAX(mb.BaseModel):
    def __init__(self, *args, name='MachineAndProcessModel_ARMAX',
                 p=2, d=0, q=2, seasonal_order=(0, 0, 0, 0),
                 **kwargs):
        super(MachineAndProcessARMAX, self).__init__(*args, **kwargs)
        self.name = name

        # ARMAX Parameter (p, d, q)
        self.p = p  # Autoregressive order
        self.d = d  # Differencing order
        self.q = q  # Moving average order
        self.seasonal_order = seasonal_order  # Seasonal ARIMA parameters

        self.model_armax = None
        self.fitted_model = None
        self.is_fitted = False

        # Für exogene Variablen
        self.feature_names = None

    def criterion(self, y_target, y_pred):
        # Behandle NaN-Werte
        mask = ~(np.isnan(y_target) | np.isnan(y_pred))
        if np.sum(mask) == 0:
            return float('inf')
        return mean_squared_error(y_target[mask], y_pred[mask])

    def prepare_data(self, X, y=None):
        """Bereite Daten für ARMAX vor"""
        if isinstance(X, pd.DataFrame):
            X_clean = X.dropna()
        else:
            X_clean = pd.DataFrame(X).dropna()

        if y is not None:
            if isinstance(y, pd.Series):
                y_clean = y.dropna()
            else:
                # Behandle verschiedene y-Formate
                if hasattr(y, 'values'):
                    y_values = y.values
                else:
                    y_values = y

                # Flatten wenn nötig (2D -> 1D)
                if isinstance(y_values, np.ndarray) and y_values.ndim > 1:
                    y_values = y_values.flatten()

                # Erstelle Series mit dem ursprünglichen Index falls vorhanden
                if hasattr(y, 'index'):
                    y_clean = pd.Series(y_values, index=y.index).dropna()
                else:
                    y_clean = pd.Series(y_values).dropna()

            # Übereinstimmende Indizes finden
            common_idx = X_clean.index.intersection(y_clean.index)
            X_clean = X_clean.loc[common_idx]
            y_clean = y_clean.loc[common_idx]

            return X_clean, y_clean

        return X_clean

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_clean = self.prepare_data(X)

        if len(X_clean) == 0:
            return np.array([])

        try:
            # Für SARIMAX mit exogenen Variablen
            if hasattr(self.fitted_model, 'forecast'):
                # Vorhersage mit exogenen Variablen
                y_pred = self.fitted_model.forecast(steps=len(X_clean), exog=X_clean.values)
            else:
                # Fallback für einfache Vorhersage
                y_pred = self.fitted_model.predict(start=0, end=len(X_clean) - 1, exog=X_clean.values)

            return np.array(y_pred)

        except Exception as e:
            print(f"Prediction error: {str(e)}")
            # Fallback: Verwende den Mittelwert der Trainingsdaten
            return np.full(len(X_clean), np.mean(self.y_train_mean) if hasattr(self, 'y_train_mean') else 0.0)

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=1, trial=None, draw_loss=False, n_outlier=12,
                    patience=10):
        best_val_error = float('inf')
        patience_counter = 0

        if isinstance(X_train, list):
            X_train = pd.concat(X_train, ignore_index=True)
            y_train = pd.concat(y_train, ignore_index=True)
        if isinstance(X_val, list):
            X_val = pd.concat(X_val, ignore_index=True)
            y_val = pd.concat(y_val, ignore_index=True)

        # Daten vorbereiten
        X_train_clean, y_train_clean = self.prepare_data(X_train, y_train)
        X_val_clean, y_val_clean = self.prepare_data(X_val, y_val)

        if len(X_train_clean) == 0 or len(y_train_clean) == 0:
            print(f'{self.name}: No valid training data available')
            return float('inf')

        # Speichere Trainingsdaten-Statistiken für Fallback
        self.y_train_mean = np.mean(y_train_clean)
        self.feature_names = X_train_clean.columns.tolist()

        if trial is None:
            n_epochs = 1

        for epoch in range(n_epochs):
            try:
                # SARIMAX Modell mit exogenen Variablen
                self.model_armax = SARIMAX(
                    endog=y_train_clean,
                    exog=X_train_clean,
                    order=(self.p, self.d, self.q),
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                # Modell fitten
                self.fitted_model = self.model_armax.fit(disp=False, maxiter=100)
                self.is_fitted = True

                # Validation
                if len(X_val_clean) > 0 and len(y_val_clean) > 0:
                    try:
                        y_val_pred = self.predict(X_val_clean)

                        # Dimensionen anpassen
                        if len(y_val_pred) != len(y_val_clean):
                            min_len = min(len(y_val_pred), len(y_val_clean))
                            y_val_pred = y_val_pred[:min_len]
                            y_val_clean = y_val_clean.iloc[:min_len]

                        val_error = self.criterion(y_val_clean.values, y_val_pred)
                    except Exception as pred_error:
                        print(f'{self.name}: Prediction error during validation: {str(pred_error)}')
                        val_error = float('inf')
                else:
                    val_error = float('inf')

                if trial:
                    trial.report(val_error, step=epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                if val_error < best_val_error:
                    best_val_error = val_error
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'{self.name}: Early stopping at epoch {epoch + 1}')
                        break

                if draw_loss:
                    plt.plot(epoch, val_error, 'bo-', label='Validation Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.show()

                print(f'{self.name}: Epoch {epoch + 1}/{n_epochs}, Val Error: {val_error:.4f}')

            except Exception as e:
                print(f'{self.name}: Error in epoch {epoch + 1}: {str(e)}')
                val_error = float('inf')
                if trial:
                    trial.report(val_error, step=epoch)

        return best_val_error

    def test_model(self, X, y_target, criterion_test=None):
        if criterion_test is None:
            criterion_test = self.criterion

        # Daten bereinigen
        X_clean, y_target_clean = self.prepare_data(X, y_target)

        if len(X_clean) == 0 or len(y_target_clean) == 0:
            return float('inf'), np.array([])

        try:
            y_pred = self.predict(X_clean)

            # Dimensionen anpassen
            if len(y_pred) != len(y_target_clean):
                min_len = min(len(y_pred), len(y_target_clean))
                y_pred = y_pred[:min_len]
                y_target_clean = y_target_clean.iloc[:min_len]

            loss = criterion_test(y_target_clean.values, y_pred)
            return loss, y_pred

        except Exception as e:
            print(f'{self.name}: Test error: {str(e)}')
            return float('inf'), np.array([])

    def get_documentation(self):
        documentation = {
            "model_type": "ARMAX (using SARIMAX)",
            "hyperparameters": {
                "p": self.p,
                "d": self.d,
                "q": self.q,
                "seasonal_order": self.seasonal_order
            },
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names
        }

        if self.fitted_model is not None:
            try:
                documentation["model_summary"] = str(self.fitted_model.summary())
            except:
                documentation["model_summary"] = "Summary not available"

        return documentation

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 500
    NUMBEROFMODELS = 1
    window_size = 1
    past_values = 0
    future_values = 0

    dataclass2 = hdata.Combined_Plate
    dataclass2.target_channels = ['curr_x']
    dataClasses = [dataclass2]

    for dataclass in dataClasses:
        dataclass.remove_bias = False
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values

    # Verwende das neue ARMAX-Modell anstelle des ursprünglichen
    model = MachineAndProcessARMAX(p=2, d=1, q=2)
    models = [model]

    hexp.run_experiment(dataClasses, use_nn_reference=False, use_rf_reference=True, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        window_size=window_size, past_values=past_values, future_values=future_values, n_drop_values=25,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name='Addition_ARMAX')