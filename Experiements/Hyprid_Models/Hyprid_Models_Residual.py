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
    def __init__(self, physical_model : mb.BaseModel=mphys.ModelErd(), ml_model : mb.BaseModel=mnn.Net(), name="Hybrid_Model"):
        super(HybridModelResidual, self).__init__(name=name +'_Phys_' + physical_model.name +'_ML_' + ml_model.name)
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

    dataSets = [hdata.DataClass_ST_Plate_Notch]

    model_phys = mphys.ModelErd()
    model_rf = mrf.RandomForestModel()
    model_nn = mnn.Net()
    model_hybrid_nn = HybridModelResidual(physical_model=model_phys, ml_model=model_nn)
    model_hybrid_rf = HybridModelResidual(physical_model=model_phys, ml_model=model_rf)

    models = [model_phys, model_nn, model_rf, model_hybrid_nn, model_hybrid_rf] # , model_net, model_rf
    # Run the experiment
    hexp.run_experiment(dataSets, models=models, NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, experiment_name=model_phys.name)