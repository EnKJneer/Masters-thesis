import copy

import numpy as np
import pandas as pd
import torch
from numpy.f2py.auxfuncs import throw_error
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_base as mb
import Models.model_neural_net as mnn
import Models.model_physical as mphys
import Models.model_random_forest as mrf
from sklearn.preprocessing import StandardScaler

class HybridModelResidual(mb.BaseModel):
    def __init__(self, physical_model : mb.BaseModel=mphys.ModelErd(), ml_model : mb.BaseModel=mnn.Net(), name=None):
        if name is None:
            name = name +'_Phys_' + physical_model.name +'_ML_' + ml_model.name
        super(HybridModelResidual, self).__init__(name=name)
        self.physical_model = physical_model
        self.ml_model = ml_model

    def predict(self, x):
        # Vorhersage des physikalischen Modells
        y_phys = self.physical_model.predict(x)
        y_corr = y_phys + self.ml_model.predict(x)

        return y_corr

    def criterion(self, y_target, y_pred):
        return np.mean(np.abs(y_target.squeeze() - y_pred.squeeze()))

    def compute_residuals(self, x, y):
        y_phys = self.physical_model.predict(x)
        residuals = y.squeeze() - y_phys
        return residuals

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        # 1. Trainiere physikalisches Modell
        print("==> Trainiere physikalisches Modell...")
        self.physical_model.train_model(X_train, y_train, X_val, y_val, **kwargs)

        # 2. Berechne Residuen
        print("==> Berechne Residuen für neuronales Netz...")

        y_train_res = self.compute_residuals(X_train, y_train)
        y_val_res = self.compute_residuals(X_val, y_val)

        # 3. Trainiere das neuronale Netz auf die Residuen
        print("==> Trainiere ML-Modell auf Residuen...")
        return self.ml_model.train_model(X_train, y_train_res, X_val, y_val_res, **kwargs)

    def test_model(self, X, y_target):
        y_pred = self.predict(X)
        loss = self.criterion(y_target, y_pred)
        return loss.item(), y_pred

    def get_documentation(self):
        return {
            "model": "HybridModel",
            "physical_model": self.physical_model.get_documentation(),
            "net_model": self.ml_model.get_documentation(),
        }
    def reset_hyperparameter(self):
        throw_error('Not implemented')

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 100
    NUMBEROFMODELS = 10

    window_size = 1
    past_values = 0
    future_values = 0

    dataSet_1 = copy.deepcopy(hdata.DataClass_ST_Plate_Notch)
    dataSet_1.name = 'Normal'
    dataSet_2 = copy.deepcopy(hdata.DataClass_ST_Plate_Notch)
    dataSet_2.add_sign_hold = True
    dataSet_2.name = 'Sign_Hold'
    dataSet_3 = copy.deepcopy(hdata.DataClass_ST_Plate_Notch)
    dataSet_3.add_sign_hold = True
    dataSet_3.name = 'Sign_Hold_x'
    dataSet_3.header = ["v_x", "a_x", "f_x_sim", "materialremoved_sim"]

    dataSets = [dataSet_1, dataSet_2, dataSet_3]

    for dataSet in dataSets:
        dataSet.add_padding = True

    model_rf = mrf.RandomForestModel(n_estimators= 100, max_depth=100, min_samples_split= 2,
                    min_samples_leaf= 4)

    model_rnn = mnn.RNN(learning_rate= 0.1, n_hidden_size= 71, n_hidden_layers= 1,
                    activation= 'ELU', optimizer_type= 'quasi_newton')

    use_rf = True
    if use_rf:
        models = [model_rf]
        postfix = 'RF_x_pinn'
    else:
        models = [model_rnn]
        postfix = 'RNN_x_pinn'

    # Run the experiment
    hexp.run_experiment(dataSets, models=models, NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        experiment_name='Pi_Feature_'+postfix)