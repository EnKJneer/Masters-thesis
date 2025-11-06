import copy
from collections import deque

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
import Helper.handling_experiment as hexp
import Helper.handling_data as hdata
import Models.model_base as mb
import Models.model_physical as mphys
import Models.model_random_forest as mrf
import Models.model_neural_net as mnn

class Experts_2(mb.BaseModel):
    def __init__(self, threshold_v_axis=0.1, name='RF_Experts_2', target_channel='curr_x', **kwargs):
        """
        Initialisiert die RF_Experts-Klasse mit drei Random Forest-Modellen.

        Parameter
        ----------
        threshold_mrr : float, optional
            Schwellenwert für MRR. Standardmäßig 100.
        threshold_v_axis : float, optional
            Schwellenwert für v_axis. Standardmäßig 0.1.
        name : str, optional
            Name des Modells. Standardmäßig 'RF_Experts'.
        """
        super().__init__(name, **kwargs)
        self.threshold_v_axis = threshold_v_axis
        self.target_channel = target_channel

        # Initialisiere die drei Experten
        self.expert1 = mrf.RandomForestModel(name="Expert_MRR_High_V_Axis_High")
        self.expert2 = mrf.RandomForestModel(name="Expert_MRR_High_V_Axis_Low")

    def _get_mask(self, X):
        axis = self.target_channel.replace('curr_', '')

        postfix = '_1_current'
        # Annahme: 'MRR' und 'v_axis' sind Spalten in X
        mask1 = (abs(X[f'v_{axis}{postfix}']) > self.threshold_v_axis)

        return mask1,  np.logical_not(mask1)

    def _split_data(self, X, y):
        """
        Teilt die Daten in drei Gruppen basierend auf den Bedingungen für MRR und v_axis.

        Parameter
        ----------
        X : pd.DataFrame oder Liste von pd.DataFrame
            Eingabedaten.
        y : pd.DataFrame oder Liste von pd.DataFrame
            Zieldaten.

        Rückgabe
        -------
        tuple
            (X1, y1), (X2, y2), (X3, y3) für die drei Experten.
        """
        if isinstance(X, list):
            X = pd.concat(X, ignore_index=True)
            y = pd.concat(y, ignore_index=True)

        mask1, mask2 = self._get_mask(X)

        X1, y1 = X[mask1], y[mask1]
        X2, y2 = X[mask2], y[mask2]

        return (X1, y1), (X2, y2)

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=1, trial=None, draw_loss=False, **kwargs):
        """
        Trainiert die drei Experten mit den entsprechenden Daten.

        Parameter
        ----------
        X_train : pd.DataFrame oder Liste von pd.DataFrame
            Trainingsdaten.
        y_train : pd.DataFrame oder Liste von pd.DataFrame
            Trainingsziele.
        X_val : pd.DataFrame oder Liste von pd.DataFrame
            Validierungsdaten.
        y_val : pd.DataFrame oder Liste von pd.DataFrame
            Validierungsziele.
        n_epochs : int, optional
            Anzahl der Epochen. Standardmäßig 1.
        trial : optuna.trial.Trial, optional
            Optuna-Trial-Objekt für Pruning.
        draw_loss : bool, optional
            Ob der Verlust gezeichnet werden soll. Standardmäßig False.

        Rückgabe
        -------
        float
            Validierungsfehler.
        """

        (X_train1, y_train1), (X_train2, y_train2) = self._split_data(X_train, y_train)
        (X_val1, y_val1), (X_val2, y_val2)= self._split_data(X_val, y_val)

        val_error1 = self.expert1.train_model(X_train1, y_train1, X_val1, y_val1, n_epochs=n_epochs, trial=trial, draw_loss=draw_loss, **kwargs)
        val_error2 = self.expert2.train_model(X_train=X_train2, y_train=y_train2,
                                              X_val=X_val2, y_val=y_val2,
                                              n_epochs=n_epochs, trial=trial, draw_loss=draw_loss, **kwargs)

        return val_error1 + val_error2

    def predict(self, X):
        """
        Vorhersage mit den drei Experten.

        Parameter
        ----------
        X : pd.DataFrame oder Liste von pd.DataFrame
            Eingabedaten.

        Rückgabe
        -------
        numpy.ndarray
            Vorhersagen.
        """
        if isinstance(X, list):
            X = pd.concat(X, ignore_index=True)

        mask1, mask2 = self._get_mask(X)

        y_pred = np.zeros(len(X))

        y_pred[mask1] = self.expert1.predict(X[mask1])
        y_pred[mask2] = self.expert2.predict(X[mask2])

        return y_pred

    def test_model(self, X, y_target):
        """
        Testet die drei Experten mit den Testdaten.

        Parameter
        ----------
        X : pd.DataFrame oder Liste von pd.DataFrame
            Testdaten.
        y_target : pd.DataFrame oder Liste von pd.DataFrame
            Testziele.

        Rückgabe
        -------
        tuple
            Verlust und Vorhersagen.
        """
        y_pred = self.predict(X)
        loss = mean_squared_error(y_target, y_pred)
        return loss, y_pred

    def criterion(self, y_target, y_pred):
        """
        Berechnet den Mean Squared Error (MSE) zwischen den Ziel- und vorhergesagten Werten.

        Parameter
        ----------
        y_target : array-like
            Zielwerte.
        y_pred : array-like
            Vorhergesagte Werte.

        Rückgabe
        -------
        float
            Berechneter MSE.
        """
        return mean_squared_error(y_target, y_pred)

    def get_documentation(self):
        """
        Gibt die Dokumentation der Hyperparameter zurück.

        Rückgabe
        -------
        dict
            Dokumentation der Hyperparameter.
        """
        documentation = {
            "threshold_v_axis": self.threshold_v_axis,
            f"expert1 {self.expert1.name}": self.expert1.get_documentation(),
            f"expert2 {self.expert2.name}": self.expert2.get_documentation(),
        }
        return documentation

    def reset_hyperparameter(self, **kwargs):
        """
        Setzt die Hyperparameter der Experten zurück.
        Parameter mit den Endungen e1, e2 oder e3 werden entsprechend aufgeteilt.
        """
        # Initialisiere leere Dictionaries für jeden Experten
        params_e1 = {}
        params_e2 = {}

        # Durchlaufe alle übergebenen Parameter
        for key, value in kwargs.items():
            if key.endswith('_e1'):
                # Entferne die Endung und füge den Parameter zu expert1 hinzu
                new_key = key[:-3]
                params_e1[new_key] = value
            elif key.endswith('_e2'):
                # Entferne die Endung und füge den Parameter zu expert2 hinzu
                new_key = key[:-3]
                params_e2[new_key] = value
            else:
                # Falls keine Endung vorhanden ist, füge den Parameter allen Experten hinzu
                params_e1[key] = value
                params_e2[key] = value

        # Setze die Hyperparameter für jeden Experten
        self.expert1.reset_hyperparameter(**params_e1)
        self.expert2.reset_hyperparameter(**params_e2)

    @staticmethod
    def get_reference_model(input_size=None):
        return Experts_2()

class EmpiricLinearModel(mb.BaseModel):
    def __init__(self, name="EmpiricLinearModel",
                 f_s = 0, a_x = 0, a_sp = 0, b = 0, f_c = 0, sigma_2 = 0,a_b =0,
                 velocity_threshold=1e-1, target_channel = 'curr_x'):
        self.name = name
        self.velocity_threshold = velocity_threshold
        self.F_s = f_s
        self.theta_f = a_x
        self.a_sp = a_sp
        self.b = b
        self.F_c = f_c
        self.sigma_2 = sigma_2
        self.theta_a = a_b
        self.target_channel = target_channel

    def sign_hold(self, v, eps=1e-1, n=3, init=-1):
        # Initialisierung des Arrays z mit Nullen
        z = np.zeros(len(v))
        h_init = np.ones(n) * init

        assert n > 1

        # Initialisierung des FiFo h mit Länge 5 und Initialwerten 0
        h = deque(h_init, maxlen=n)

        # Berechnung von z
        for i in range(len(v)):
            if abs(v[i]) > eps:
                h.append(v[i])

            if i >= n - 1:  # Da wir ab dem 5. Element starten wollen
                # Berechne zi als Vorzeichen der Summe
                z[i] = np.sign(sum(h))

        return z

    def criterion(self, y_target, y_pred):
        return np.mean(np.abs(y_target - y_pred))

    def predict(self, X):
        for target in self.target_channel:
            axis = target.replace('curr_', '')
            v_x = X[f'v_{axis}_1_current'].values
            a_x = X[f'a_{axis}_1_current'].values
            f_x_sim = X[f'f_{axis}_1_current'].values

            #v_sp = X[f'v_sp_1_current'].values

            stillstand_mask = (np.abs(v_x) <= self.velocity_threshold)
            bewegung_mask = ~stillstand_mask

            y_pred = np.zeros_like(v_x, dtype=float)
            v_s = self.sign_hold(v_x)
            y_pred[stillstand_mask] = (self.F_s * v_s[stillstand_mask] +
                                       self.theta_f * f_x_sim[stillstand_mask] +
                                       self.b ) #* v_sp[stillstand_mask]

            y_pred[bewegung_mask] = (self.F_c * np.sign(v_x[bewegung_mask]) +
                                     self.sigma_2 * v_x[bewegung_mask] +
                                     self.theta_f * f_x_sim[bewegung_mask] +
                                     self.theta_a * a_x[bewegung_mask] +
                                     self.b) # * v_sp[bewegung_mask]

        return y_pred

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        data = X_train.copy()
        data[self.target_channel] = y_train.values

        params, _, _ = self.fit_model(data)

        self.F_s = params['F_s']
        self.theta_f = params['theta_f']
        self.b = params['b']
        self.F_c = params['F_c']
        self.sigma_2 = params['sigma_2']
        self.theta_a = params['theta_a']

        # Ausgabe der Parameter
        for param_name, param_value in params.items():
            print(f"{param_name}: {param_value}")

        validation_loss = self.criterion(y_val.values.squeeze(), self.predict(X_val))
        print(f"Validation Loss: {validation_loss:.4e}")
        return validation_loss

    def fit_model(self, data):
        if type(self.target_channel) is not list:
            self.target_channel = [self.target_channel]
        for target in self.target_channel:
            axis = target.replace('curr_', '')
            v_x = data[f'v_{axis}_1_current'].values
            a_x = data[f'a_{axis}_1_current'].values
            f_x_sim = data[f'f_{axis}_1_current'].values
            v_s = self.sign_hold(v_x)

            #v_sp = data[f'v_sp_1_current'].values

            curr_x = data[target].values

            stillstand_mask = (np.abs(v_x) <= self.velocity_threshold)
            bewegung_mask = ~stillstand_mask

            params = {}

            if np.sum(stillstand_mask) > 2:
                X_stillstand = np.column_stack([v_s[stillstand_mask],
                                                f_x_sim[stillstand_mask],
                                                np.ones(np.sum(stillstand_mask))]) # v_sp[stillstand_mask]
                y_stillstand = curr_x[stillstand_mask]
                reg_stillstand = LinearRegression(fit_intercept=False)
                reg_stillstand.fit(X_stillstand, y_stillstand)
                params['F_s'] = reg_stillstand.coef_[0]
                params['theta_f'] = reg_stillstand.coef_[1]
                params['b'] = reg_stillstand.coef_[2]
            else:
                print("Warnung: Nicht genügend Stillstandspunkte für Fitting")
                params['F_s'] = 0
                params['theta_f'] = 1
                params['b'] = 0

            if np.sum(bewegung_mask) > 4:
                X_bewegung = np.column_stack([np.sign(v_x[bewegung_mask]), v_x[bewegung_mask],
                                              f_x_sim[bewegung_mask],
                                              a_x[bewegung_mask], np.ones(np.sum(bewegung_mask))]) #v_sp[bewegung_mask]
                y_bewegung = curr_x[bewegung_mask]
                reg_bewegung = LinearRegression(fit_intercept=False)
                reg_bewegung.fit(X_bewegung, y_bewegung)
                params['F_c'] = reg_bewegung.coef_[0]
                params['sigma_2'] = reg_bewegung.coef_[1]
                params['theta_f'] = reg_bewegung.coef_[2]
                params['theta_a'] = reg_bewegung.coef_[3]
                params['b'] = reg_bewegung.coef_[4]
            else:
                print("Warnung: Nicht genügend Bewegungspunkte für Fitting")
                params['F_c'] = 0
                params['sigma_2'] = 0
                params['theta_f'] = 1
                params['a_y'] = 1
                params['a_z'] = 1
                params['a_sp'] = 0
                params['theta_a'] = 0
                params['b_s'] = 0

        return params, stillstand_mask, bewegung_mask

    def test_model(self, X, y_target):
        prediction = self.predict(X)
        loss = self.criterion(y_target.values.squeeze(), prediction)
        return loss, prediction

    def get_documentation(self):
        documentation = {
            "description": "This model fits a two-stage friction model: steady and movement.",
            "parameters": {
                "name": self.name,
                "velocity_threshold": self.velocity_threshold,
                "F_s": self.F_s,
                "a_x": self.theta_f,
                "a_sp": self.a_sp,
                "b": self.b,
                "F_c": self.F_c,
                "sigma_2": self.sigma_2,
                "a_b": self.theta_a,
            }
        }
        return documentation
    def reset_hyperparameter(self, **kwargs):
        """
        Resets the hyperparameters of the friction model.
        Only updates parameters that are explicitly provided in kwargs.
        """
        # Update only if parameter exists in kwargs
        if 'f_s' in kwargs:
            self.F_s = kwargs['f_s']
        if 'a_x' in kwargs:
            self.theta_f = kwargs['a_x']
        if 'a_sp' in kwargs:
            self.a_sp = kwargs['a_sp']
        if 'b' in kwargs:
            self.b = kwargs['b']
        if 'f_c' in kwargs:
            self.F_c = kwargs['f_c']
        if 'sigma_2' in kwargs:
            self.sigma_2 = kwargs['sigma_2']
        if 'a_b' in kwargs:
            self.theta_a = kwargs['a_b']
        if 'velocity_threshold' in kwargs:
            self.velocity_threshold = kwargs['velocity_threshold']
        if 'target_channel' in kwargs:
            self.target_channel = kwargs['target_channel']

if __name__ == '__main__':

    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataSet = hdata.DataClass_ST_Plate_Notch_Mes

    dataclass1 = copy.copy(dataSet)
    axis = 'x'
    dataclass1.target_channels = [f'curr_{axis}']
    dataclass1.header = [f"v_{axis}", f"a_{axis}", f"f_{axis}", "materialremoved_sim"] # ["pos_x", "v_x", "a_x", "f_x_sim", "materialremoved_sim"]

    dataClasses = [dataclass1]
    for dataclass in dataClasses:
        dataclass.add_padding = True
        dataclass.add_sign_hold = True

    model_rf = mrf.RandomForestModel(n_estimators=100, max_depth=100, min_samples_split=2,
                                     min_samples_leaf=4)

    model_rnn = mnn.RNN(learning_rate=0.1, n_hidden_size=71, n_hidden_layers=1,
                        activation='ELU', optimizer_type='quasi_newton')

    model_phys = EmpiricLinearModel()
    model_phys.target_channel = dataclass1.target_channels

    model = Experts_2(threshold_v_axis=1)
    model.expert1 = copy.deepcopy(model_phys)
    model.expert2 = copy.deepcopy(model_rnn)
    model.target_channel = dataclass1.target_channels

    model.name = 'Mixed_Experts_2'
    models = [model,model_rnn,model_phys]

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'model_heatmap', 'prediction_overview'], experiment_name='Mes_'+model.name+'_'+axis)
