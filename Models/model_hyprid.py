import numpy as np
import optuna
import pandas as pd
import torch
import torch.jit
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
import Models.model_base as mb
import Models.model_neural_net as mnn
import Models.model_physical as mphys

class HybridModelResidual(mb.BaseNetModel):
    def __init__(self, physical_model=mphys.PhysicalModelErdSingleAxis(0.01, 0.01, 0.01, 0.01, 0.01, learning_rate=1), net_model=mnn.Net(), name="Hybrid_Model"):
        super(HybridModelResidual, self).__init__()
        self.physical_model = physical_model
        self.net_model = net_model
        self.name = name+'_Phys_'+physical_model.name+'_ML_'+net_model.name

        # Überprüfen, ob physical_model die Methode .to() hat
        has_to_physical = hasattr(self.physical_model, 'to')

        # Überprüfen, ob net_model die Methode .to() hat
        has_to_net = hasattr(self.net_model, 'to')
        if has_to_physical and has_to_net:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(self.device)
            self.physical_model.to(self.device)
            self.net_model.to(self.device)
        else:
            self.device = torch.device('cpu')
            self.to(self.device)

    def forward(self, x):
        # Vorhersage des physikalischen Modells
        with torch.no_grad():
            #x_df = pd.DataFrame(self.scaler.inverse_transform(x.cpu().numpy()), columns=self.original_columns)
            phys_input = self.physical_model.get_input_vector(x) # x_df
            y_phys = self.physical_model.model(phys_input)

        if self.physical_model.device == self.net_model.device and type(y_phys) == torch.Tensor:
            # Korrigierte Vorhersage = physikalisch + residuum
            y_corr = y_phys.squeeze() + self.net_model.predict(x).squeeze()
        else:
            y_corr = y_phys.cpu().squeeze() + self.net_model.predict(x).squeeze()
        return y_corr

    def _initialize(self):
        self.net_model._initialize()
        self.physical_model.reset_parameters()

    def scale_data(self, X):
        # Skaliere wie üblich über das neuronale Netz
        if hasattr(self.net_model, 'scaler') and self.net_model.scaler is not None:
            self.scaler = self.net_model.scaler
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            self.net_model.scaler = self.scaler

        self.original_columns = X.columns if isinstance(X, pd.DataFrame) else None
        return self.scaler.transform(X)

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        print(f"[HybridModel] Device: {self.device}")
        #print(f"[PhysicalModel] Device: {(self.physical_model.parameters()).device}")
        #print(f"[NetModel] Device: {(self.net_model.parameters()).device}")

        # 1. Trainiere physikalisches Modell
        print("==> Trainiere physikalisches Modell...")
        self.physical_model.train_model(X_train, y_train, X_val, y_val, **kwargs)

        # 2. Berechne Residuen
        print("==> Berechne Residuen für neuronales Netz...")

        def compute_residuals(X, y):
            residuals = []
            for x_batch, y_batch in zip(X, y) if isinstance(X, list) else [(X, y)]:
                with torch.no_grad():
                    input_vector = self.physical_model.get_input_vector(x_batch)
                    y_phys = self.physical_model.model(input_vector)
                    if not isinstance(y_batch, torch.Tensor):
                        y_batch = torch.tensor(y_batch.values if hasattr(y_batch, 'values') else y_batch,
                                               dtype=torch.float32).to(self.physical_model.device)
                    res = y_batch.squeeze() - y_phys
                    residuals.append(res.detach().cpu().numpy())
            return residuals if isinstance(X, list) else residuals[0]

        y_train_res = compute_residuals(X_train, y_train)
        y_val_res = compute_residuals(X_val, y_val)

        # 3. Trainiere das neuronale Netz auf die Residuen
        print("==> Trainiere neuronales Netz auf Residuen...")
        return self.net_model.train_model(X_train, y_train_res, X_val, y_val_res, **kwargs)

    def predict(self, X):
        return self.forward(X)

    def test_model(self, X, y_target):
        if not isinstance(y_target, np.ndarray):
            y_target = y_target.to_numpy()
        y_target = torch.tensor(y_target, dtype=torch.float32).to(self.device)
        y_pred = self.predict(X)
        loss = self.criterion(y_target, y_pred)
        return loss.item(), y_pred.detach().cpu().numpy()

    def get_documentation(self):
        return {
            "model": "HybridModel",
            "physical_model": self.physical_model.get_documentation(),
            "net_model": self.net_model.get_documentation(),
        }

class HybridModelGuidedML(mb.BaseNetModel):
    def __init__(self, physical_model=mphys.PhysicalModelErdSingleAxis(0.01, 0.01, 0.01, 0.01, 0.01, learning_rate=1), net_model=mnn.Net(), name="Hybrid_Model"):
        super(HybridModelGuidedML, self).__init__()
        self.physical_model = physical_model
        self.net_model = net_model
        self.name = name+'_Phys_'+physical_model.name+'_ML_'+net_model.name

        # Überprüfen, ob physical_model die Methode .to() hat
        has_to_physical = hasattr(self.physical_model, 'to')

        # Überprüfen, ob net_model die Methode .to() hat
        has_to_net = hasattr(self.net_model, 'to')
        if has_to_physical and has_to_net:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(self.device)
            self.physical_model.to(self.device)
            self.net_model.to(self.device)
        else:
            self.device = torch.device('cpu')
            self.to(self.device)

    def forward(self, x):
        # Vorhersage des physikalischen Modells
        with torch.no_grad():
            phys_input = self.physical_model.get_input_vector(x)
            y_phys = self.physical_model.model(phys_input)

        # Kombiniere die Eingabe mit der Vorhersage des physikalischen Modells
        x['phy_model'] = y_phys.cpu().squeeze()
        combined_input = x.values
        y_corr = self.net_model.predict(combined_input)
        return y_corr

    def _initialize(self):
        self.net_model._initialize()
        self.physical_model.reset_parameters()

    def scale_data(self, X):
        # Skaliere wie üblich über das neuronale Netz
        if hasattr(self.net_model, 'scaler') and self.net_model.scaler is not None:
            self.scaler = self.net_model.scaler
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            self.net_model.scaler = self.scaler

        self.original_columns = X.columns if isinstance(X, pd.DataFrame) else None
        return self.scaler.transform(X)

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        print(f"[HybridModel] Device: {self.device}")

        # 1. Trainiere physikalisches Modell
        print("==> Trainiere physikalisches Modell...")
        self.physical_model.train_model(X_train, y_train, X_val, y_val, **kwargs)

        # 2. Berechne Vorhersagen des physikalischen Modells für Trainings- und Validierungsdaten
        print("==> Berechne Vorhersagen des physikalischen Modells...")

        def compute_physical_predictions(X):
            predictions = []
            for x_batch in X if isinstance(X, list) else [X]:
                with torch.no_grad():
                    input_vector = self.physical_model.get_input_vector(x_batch)
                    y_phys = self.physical_model.model(input_vector)
                    predictions.append(y_phys.detach().cpu().numpy())
            return predictions if isinstance(X, list) else predictions[0]

        y_train_phys = compute_physical_predictions(X_train)
        y_val_phys = compute_physical_predictions(X_val)

        # 3. Kombiniere die Eingabe mit den Vorhersagen des physikalischen Modells
        y_train_phys = y_train_phys.reshape(-1, 1)  # Umformen in ein 2D-Array
        X_train_combined = np.concatenate((X_train, y_train_phys), axis=1)
        y_val_phys = y_val_phys.reshape(-1, 1)  # Umformen in ein 2D-Array
        X_val_combined = np.concatenate((X_val, y_val_phys), axis=1)

        # 4. Trainiere das neuronale Netz auf die kombinierten Eingaben
        print("==> Trainiere neuronales Netz auf kombinierte Eingaben...")
        return self.net_model.train_model(X_train_combined, y_train, X_val_combined, y_val, **kwargs)

    def predict(self, X):
        return self.forward(X)

    def test_model(self, X, y_target):
        if not isinstance(y_target, np.ndarray):
            y_target = y_target.to_numpy()
        y_target = torch.tensor(y_target, dtype=torch.float32).to(self.device)
        y_pred = self.predict(X)
        if type(y_target) == torch.Tensor and type(y_pred) is not torch.Tensor:
            y_pred = torch.tensor(y_pred, dtype=torch.float32).to(self.device)
            loss = self.criterion(y_target, y_pred)
            return loss.item(), y_pred.detach().cpu().numpy()
        else:
            loss = self.criterion(y_target, y_pred)
            return loss.item(), y_pred.detach().cpu().numpy()

    def get_documentation(self):
        return {
            "model": "HybridModel",
            "physical_model": self.physical_model.get_documentation(),
            "net_model": self.net_model.get_documentation(),
        }