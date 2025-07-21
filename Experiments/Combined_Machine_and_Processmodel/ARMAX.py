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
import Models.model_random_forest as mrf
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import optuna  # Optional, falls du Optuna für Trials verwendest
from sklearn.neural_network import MLPRegressor

class ARMAX_RF_CombinedModel(mb.BaseModel):
    def __init__(self, armax_params=None, rf_params=None, name="ARMAX_RF_CombinedModel"):
        super().__init__()
        self.name = name
        self.armax_model = ARMAXModel(**(armax_params or {}))
        self.rf_model = mrf.RandomForestModel(**(rf_params or {}))
        self.is_fitted = False

    def criterion(self, y_true, y_pred):
        # Beispiel: MSE über RF-Vorhersage
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(y_true, y_pred)

    def get_documentation(self):
        return {
            "name": self.name,
            "armax_model": self.armax_model.get_documentation(),
            "random_forest_model": self.rf_model.get_documentation()
        }

    def prepare_data_armax(self, X):
        cols = [col for col in X.columns if col.startswith('a_') or col.startswith('v_')]
        return X[cols]

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        X_train_armax = self.prepare_data_armax(X_train)
        X_val_armax = self.prepare_data_armax(X_val)

        print(f"{self.name}: Training ARMAX model...")
        armax_loss = self.armax_model.train_model(X_train_armax, y_train, X_val_armax, y_val, **kwargs)

        armax_pred_train = self.armax_model.predict(X_train_armax)
        armax_pred_val = self.armax_model.predict(X_val_armax)

        X_train_rf = X_train.copy()
        X_val_rf = X_val.copy()
        X_train_rf['armax_pred'] = armax_pred_train
        X_val_rf['armax_pred'] = armax_pred_val

        print(f"{self.name}: Training Random Forest model with ARMAX predictions as additional feature...")
        rf_loss = self.rf_model.train_model(X_train_rf, y_train, X_val_rf, y_val, **kwargs)

        self.is_fitted = True
        return armax_loss, rf_loss

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError(f"{self.name}: Model must be trained before prediction")

        X_armax = self.prepare_data_armax(X)
        armax_pred = self.armax_model.predict(X_armax)

        X_rf = X.copy()
        X_rf['armax_pred'] = armax_pred

        rf_pred = self.rf_model.predict(X_rf)
        return rf_pred

    def test_model(self, X, y_target, criterion=None):
        if criterion is None:
            criterion = self.criterion

        y_pred = self.predict(X)
        loss = criterion(y_target, y_pred)
        return loss, y_pred

class ARMAXModel(mb.BaseModel):
    def __init__(self, *args, name='ARMAXModel',
                 p=2, d=0, q=2, seasonal_order=(0, 0, 0, 0),
                 **kwargs):
        super(ARMAXModel, self).__init__(*args, **kwargs)
        self.name = name
        self.p = p
        self.d = d
        self.q = q
        self.seasonal_order = seasonal_order

        self.model_armax = None
        self.fitted_model = None
        self.is_fitted = False
        self.feature_names = None
        self.y_train_mean = 0.0

    def criterion(self, y_target, y_pred):
        mask = ~(np.isnan(y_target) | np.isnan(y_pred))
        if np.sum(mask) == 0:
            return float('inf')
        return mean_squared_error(y_target[mask], y_pred[mask])

    def prepare_data(self, X, y=None):
        X_clean = pd.DataFrame(X).dropna()

        if y is not None:
            y_values = pd.Series(np.asarray(y).flatten())
            y_clean = y_values.dropna()

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
            y_pred = self.fitted_model.forecast(steps=len(X_clean), exog=X_clean.values)
            return np.array(y_pred)
        except Exception as e:
            print(f"{self.name}: Prediction error: {e}")
            return np.full(len(X_clean), self.y_train_mean)

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=1, trial=None,
                    draw_loss=False, patience=10, n_outlier=12):
        best_val_error = float('inf')
        patience_counter = 0

        if isinstance(X_train, list):
            X_train = pd.concat(X_train, ignore_index=True)
            y_train = pd.concat(y_train, ignore_index=True)
        if isinstance(X_val, list):
            X_val = pd.concat(X_val, ignore_index=True)
            y_val = pd.concat(y_val, ignore_index=True)

        X_train_clean, y_train_clean = self.prepare_data(X_train, y_train)
        X_val_clean, y_val_clean = self.prepare_data(X_val, y_val)

        if len(X_train_clean) == 0 or len(y_train_clean) == 0:
            print(f"{self.name}: No valid training data available.")
            return float('inf')

        self.y_train_mean = y_train_clean.mean()
        self.feature_names = X_train_clean.columns.tolist()

        if trial is None:
            n_epochs = 1

        for epoch in range(n_epochs):
            try:
                self.model_armax = SARIMAX(
                    endog=y_train_clean,
                    exog=X_train_clean,
                    order=(self.p, self.d, self.q),
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self.fitted_model = self.model_armax.fit(disp=False, maxiter=100)
                self.is_fitted = True

                y_val_pred = self.predict(X_val_clean)
                min_len = min(len(y_val_pred), len(y_val_clean))
                val_error = self.criterion(y_val_clean.values[:min_len], y_val_pred[:min_len])

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
                        print(f"{self.name}: Early stopping at epoch {epoch + 1}")
                        break

                if draw_loss:
                    plt.plot(epoch, val_error, 'bo-')
                    plt.xlabel("Epoch")
                    plt.ylabel("Validation Loss")
                    plt.title(f"{self.name} Validation Loss")
                    plt.show()

                print(f"{self.name}: Epoch {epoch + 1}/{n_epochs}, Val Error: {val_error:.4f}")

            except Exception as e:
                print(f"{self.name}: Training error in epoch {epoch + 1}: {e}")
                val_error = float('inf')

        return best_val_error

    def test_model(self, X, y_target, criterion_test=None):
        if criterion_test is None:
            criterion_test = self.criterion

        X_clean, y_target_clean = self.prepare_data(X, y_target)

        if len(X_clean) == 0 or len(y_target_clean) == 0:
            return float('inf'), np.array([])

        try:
            y_pred = self.predict(X_clean)
            min_len = min(len(y_pred), len(y_target_clean))
            loss = criterion_test(y_target_clean.values[:min_len], y_pred[:min_len])
            return loss, y_pred
        except Exception as e:
            print(f"{self.name}: Test error: {e}")
            return float('inf'), np.array([])

    def get_documentation(self):
        documentation = {
            "model_type": "ARMAX using SARIMAX",
            "hyperparameters": {
                "p": self.p,
                "d": self.d,
                "q": self.q,
                "seasonal_order": self.seasonal_order
            },
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names
        }
        if self.fitted_model:
            try:
                documentation["model_summary"] = str(self.fitted_model.summary())
            except Exception:
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

    dataclass2 = hdata.Combined_PlateNotch_OldData
    dataclass2.target_channels = ['curr_x']
    dataClasses = [dataclass2]

    for dataclass in dataClasses:
        dataclass.remove_bias = False
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values

    # Verwende das neue ARMAX-Modell anstelle des ursprünglichen
    model = ARMAX_RF_CombinedModel()
    model1 = ARMAXModel()
    models = [model, model1]

    hexp.run_experiment(dataClasses, use_nn_reference=False, use_rf_reference=True, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        window_size=window_size, past_values=past_values, future_values=future_values, n_drop_values=25,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name='ARMAX')