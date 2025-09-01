import copy

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
import Helper.handling_experiment as hexp
import Helper.handling_data as hdata
import Models.model_base as mb
import Models.model_physical as mphys
from Models.model_random_forest import RandomForestModel
from Models.model_neural_net import RNN

class Experts(mb.BaseModel):
    def __init__(self, threshold_mrr=100, threshold_v_axis=0.1, name='RF_Experts', target_channel = 'curr_x', **kwargs):
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
        self.threshold_mrr = threshold_mrr
        self.threshold_v_axis = threshold_v_axis
        self.target_channel = target_channel

        # Initialisiere die drei Experten
        self.expert1 = RandomForestModel(name="Expert_MRR_Low")
        self.expert2 = RandomForestModel(name="Expert_MRR_High_V_Axis_High")
        self.expert3 = RandomForestModel(name="Expert_MRR_High_V_Axis_Low")

    def _get_mask(self, X):
        axis = self.target_channel.replace('curr_', '')

        postfix = '_1_current'
        # Annahme: 'MRR' und 'v_axis' sind Spalten in X
        mask1 = X[f'materialremoved_sim{postfix}'] < self.threshold_mrr
        mask2 = (X[f'materialremoved_sim{postfix}'] >= self.threshold_mrr) & (abs(X[f'v_{axis}{postfix}']) > self.threshold_v_axis)
        mask3 = (X[f'materialremoved_sim{postfix}'] >= self.threshold_mrr) & ((X[f'v_{axis}{postfix}']) <= self.threshold_v_axis)

        return mask1, mask2, mask3

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

        mask1, mask2, mask3 = self._get_mask(X)

        X1, y1 = X[mask1], y[mask1]
        X2, y2 = X[mask2], y[mask2]
        X3, y3 = X[mask3], y[mask3]

        return (X1, y1), (X2, y2), (X3, y3)

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

        (X_train1, y_train1), (X_train2, y_train2), (X_train3, y_train3) = self._split_data(X_train, y_train)
        (X_val1, y_val1), (X_val2, y_val2), (X_val3, y_val3) = self._split_data(X_val, y_val)

        val_error1 = self.expert1.train_model(X_train1, y_train1, X_val1, y_val1, n_epochs, trial, draw_loss, **kwargs)
        val_error2 = self.expert2.train_model(X_train2, y_train2, X_val2, y_val2, n_epochs, trial, draw_loss, **kwargs)
        val_error3 = self.expert3.train_model(X_train3, y_train3, X_val3, y_val3, n_epochs, trial, draw_loss, **kwargs)

        return val_error1 + val_error2 + val_error3

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

        mask1, mask2, mask3 = self._get_mask(X)

        y_pred = np.zeros(len(X))

        y_pred[mask1] = self.expert1.predict(X[mask1])
        y_pred[mask2] = self.expert2.predict(X[mask2])
        y_pred[mask3] = self.expert3.predict(X[mask3])

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
            "threshold_mrr": self.threshold_mrr,
            "threshold_v_axis": self.threshold_v_axis,
            "expert1": self.expert1.get_documentation(),
            "expert2": self.expert2.get_documentation(),
            "expert3": self.expert3.get_documentation()
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
        params_e3 = {}

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
            elif key.endswith('_e3'):
                # Entferne die Endung und füge den Parameter zu expert3 hinzu
                new_key = key[:-3]
                params_e3[new_key] = value
            else:
                # Falls keine Endung vorhanden ist, füge den Parameter allen Experten hinzu
                params_e1[key] = value
                params_e2[key] = value
                params_e3[key] = value

        # Setze die Hyperparameter für jeden Experten
        self.expert1.reset_hyperparameter(**params_e1)
        self.expert2.reset_hyperparameter(**params_e2)
        self.expert3.reset_hyperparameter(**params_e3)

    @staticmethod
    def get_reference_model(input_size=None):
        return Experts()
if __name__ == '__main__':

    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataSet = hdata.DataClass_ST_Plate_Notch
  
    dataSet.training_data_paths =  ['S235JR_Plate_Normal_0.csv', 'S235JR_Plate_Normal_4.csv',
                                    'S235JR_Plate_Normal_1.csv', 'S235JR_Plate_Normal_2.csv',
                                    'S235JR_Plate_SF_0.csv', 'S235JR_Plate_Depth_0.csv',
                                    'S235JR_Plate_SF_1.csv', 'S235JR_Plate_Depth_1.csv',
                                    'S235JR_Plate_SF_2.csv', 'S235JR_Plate_Depth_2.csv',
                                    'S235JR_Plate_SF_3.csv', 'S235JR_Plate_Depth_3.csv',
                                    'S235JR_Plate_SF_4.csv', 'S235JR_Plate_Depth_4.csv',]
    dataSet.validation_data_paths = ['S235JR_Notch_Normal_0.csv','S235JR_Notch_Normal_1.csv',
                                     'S235JR_Notch_Normal_2.csv', 'S235JR_Notch_Normal_3.csv',
                                     'S235JR_Notch_Normal_4.csv', 'S235JR_Notch_Depth_0.csv',
                                     'S235JR_Notch_Depth_1.csv', 'S235JR_Notch_Depth_2.csv',
                                     'S235JR_Notch_Depth_3.csv', 'S235JR_Notch_Depth_4.csv',]
    dataSet.testing_data_paths = [  'AL_2007_T4_Gear_Normal_3.csv','AL_2007_T4_Plate_Normal_3.csv',
                                    'S235JR_Gear_Normal_3.csv','S235JR_Plate_Normal_3.csv']
    dataSet.header = ["v_sp", "v_x", "v_y", "v_z", "a_x", "a_y", "a_z", "a_sp", "f_x", "f_y", "f_z", "materialremoved_sim"]
    dataclass1 = copy.copy(dataSet)
    dataclass1.add_sign_hold = True

    dataClasses = [dataclass1]
    for dataclass in dataClasses:
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values
        dataclass.add_padding = True

    model_rf = RandomForestModel(n_estimators= 52,max_features= 500, min_samples_split= 67,
                    min_samples_leaf= 4)

    model_rnn = RNN(learning_rate= 0.04834201195017264, n_hidden_size= 94, n_hidden_layers= 1,
                    activation= 'Sigmoid', optimizer_type= 'quasi_newton')
    model_lin = mphys.LinearModel()
    model = Experts()
    model.expert1 = copy.deepcopy(model_rf)
    model.expert2 = copy.deepcopy(model_rf)
    model.expert3 = copy.deepcopy(model_rf)

    models = [model]

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['model_heatmap', 'prediction_overview'], experiment_name=dataSet.name)
