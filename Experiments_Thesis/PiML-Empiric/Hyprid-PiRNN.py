import copy
import datetime
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_experiment as hexp
import Models.model_random_forest as mrf
import Models.model_neural_net as mnn
import Models.model_physical as mphy
import Models.model_base as mb


class PiRNN(mb.BaseTorchModel):
    def __init__(
            self,
            input_size=None,
            output_size=1,
            n_hidden_size=None,
            n_hidden_layers=1,
            activation='ReLU',
            learning_rate=1,
            name="PiRNN",
            batched_input=False,
            optimizer_type='quasi_newton',
            dropout_rate=0.0,
            z_dim=1,
            dt=50.0,
            axis='x',
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type,
            dropout_rate=dropout_rate,
        )
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation_name = activation
        self.activation = self.activation_map[activation]()
        self.batched_input = batched_input
        self.z_dim = z_dim
        self.dt = dt
        self.axis = axis

        # Gespeicherte Indizes für a, v, F, materialremoved
        self.idx_v = None
        self.idx_a = None
        self.idx_F = None
        self.idx_materialremoved = None

        # Flag ob lineare Regression trainiert wurde
        self.linear_regression_trained = False

        # Hyperparameter für das zusätzliche NN
        self.n_nn_hidden_size = n_hidden_size
        self.n_nn_hidden_layers = n_hidden_layers

        # Initialisierung der Schichten
        if self.input_size is not None:
            self._initialize()

    def _initialize(self):
        """Initialisiert RNN, z_layer, lineare Regression, zusätzliches NN und Output-Layer."""
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size
        if self.n_nn_hidden_size is None:
            self.n_nn_hidden_size = self.input_size

        # RNN für z
        self.rnn = nn.RNN(
            1,
            self.n_hidden_size,
            self.n_hidden_layers,
            batch_first=self.batched_input,
            dropout=self.dropout_rate if self.n_hidden_layers > 1 else 0.0,
        )
        self.z_layer = nn.Linear(self.n_hidden_size, self.z_dim)

        # Lineare Regression (a, v, F -> y) ohne Bias
        self.linear_regression = nn.Linear(3, self.output_size, bias=False)

        # RNN Output Layer (z, dz_dt -> y_residual) mit Bias
        self.rnn_output = nn.Linear(2 * self.z_dim, self.output_size, bias=True)

        # Zusätzliches Feed-Forward NN (materialremoved, F -> y_nn)
        nn_layers = []
        nn_layers.append(nn.Linear(2, self.n_nn_hidden_size))
        nn_layers.append(self.activation)

        for _ in range(self.n_nn_hidden_layers - 1):
            nn_layers.append(nn.Linear(self.n_nn_hidden_size, self.n_nn_hidden_size))
            nn_layers.append(self.activation)
            if self.dropout_rate > 0:
                nn_layers.append(nn.Dropout(self.dropout_rate))

        nn_layers.append(nn.Linear(self.n_nn_hidden_size, self.output_size))
        self.ffnn = nn.Sequential(*nn_layers)

        self.to(self.device)

    def _set_physics_indices(self, X, target):
        """
        Bestimmt die Indizes von a, v, F, materialremoved im DataFrame X basierend auf target.
        """
        axis = target.replace('curr_', '')
        self.col_v = f'v_{axis}_1_current'
        self.col_a = f'a_{axis}_1_current'
        self.col_F = f'f_{axis}_sim_1_current'
        self.col_materialremoved = 'materialremoved_sim_1_current'

        if isinstance(X, pd.DataFrame):
            self.idx_v = X.columns.get_loc(self.col_v)
            self.idx_a = X.columns.get_loc(self.col_a)
            self.idx_F = X.columns.get_loc(self.col_F)
            self.idx_materialremoved = X.columns.get_loc(self.col_materialremoved)
        else:
            raise ValueError("X muss ein DataFrame sein, um die Indizes zu bestimmen!")

    def _extract_physics_terms(self, x):
        """
        Extrahiere a, v, F, materialremoved aus dem Input-Tensor x.
        """
        if isinstance(x, torch.Tensor):
            if self.idx_v is None or self.idx_a is None or self.idx_F is None:
                raise ValueError("Indizes für a, v, F wurden nicht gesetzt!")
            v = x[:, :, self.idx_v].unsqueeze(-1)
            a = x[:, :, self.idx_a].unsqueeze(-1)
            F = x[:, :, self.idx_F].unsqueeze(-1)
            materialremoved = x[:, :, self.idx_materialremoved].unsqueeze(-1)
            return a, v, F, materialremoved
        else:
            raise ValueError("Input x muss ein Tensor sein!")

    def _train_linear_regression(self, X_train, y_train):
        """
        Trainiert die lineare Regression auf gefilterten Daten.
        Filter: |v| > 1 UND materialremoved > 100
        """
        # Konvertiere zu numpy für einfachere Filterung
        if isinstance(X_train, pd.DataFrame):
            v_values = X_train[self.col_v].values
            materialremoved_values = X_train[self.col_materialremoved].values
            a_values = X_train[self.col_a].values
            F_values = X_train[self.col_F].values
            y_values = y_train.values if isinstance(y_train, pd.Series) else y_train
        else:
            raise ValueError("X_train muss ein DataFrame sein!")

        # Filter anwenden
        mask = (np.abs(v_values) > 1) & (materialremoved_values > 100)

        if np.sum(mask) == 0:
            raise ValueError("Keine Datenpunkte erfüllen die Filterkriterien!")

        # Gefilterte Daten
        a_filtered = a_values[mask]
        v_filtered = v_values[mask]
        F_filtered = F_values[mask]
        y_filtered = y_values[mask]

        # Erstelle Feature-Matrix [a, v, F]
        X_linear = np.stack([a_filtered, v_filtered, F_filtered], axis=1)

        # Trainiere mit Least Squares
        X_tensor = torch.tensor(X_linear, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_filtered.values, dtype=torch.float32, device=self.device)#.unsqueeze(-1)

        # Analytische Lösung: w = (X^T X)^{-1} X^T y
        XTX = X_tensor.T @ X_tensor
        XTy = X_tensor.T @ y_tensor
        weights = torch.linalg.solve(XTX, XTy)

        # Setze Gewichte in linear_regression
        with torch.no_grad():
            self.linear_regression.weight.copy_(weights.T)

        # Friere Parameter ein
        #self.linear_regression.requires_grad_(False)
        #self.linear_regression_trained = True

        print(f"Lineare Regression trainiert auf {np.sum(mask)} von {len(mask)} Datenpunkten")
        print(f"Gelernte Gewichte: {weights.T.cpu().numpy()}")

    def forward(self, x):
        """
        Forward-Pass: y_pred = y_linear(a,v,F) + y_rnn(z, dz_dt) + y_nn(materialremoved, F)
        """
        # Füge Batch-Dimension hinzu, falls nicht vorhanden
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = x.to(self.device)
        if not x.is_contiguous():
            x = x.contiguous()

        batch_size = x.size(1)
        seq_len = x.size(0)

        # Extrahiere a, v, F, materialremoved
        a, v, F, materialremoved = self._extract_physics_terms(x)

        # Lineare Regression: y_linear = w1*a + w2*v + w3*F
        avF = torch.cat([a, v, F], dim=-1)  # [seq_len, batch, 3]
        y_linear = self.linear_regression(avF)  # [seq_len, batch, output_size]

        # RNN-Pass für z
        h0 = torch.zeros(
            self.n_hidden_layers,
            batch_size,
            self.n_hidden_size,
            device=self.device
        )
        x_pinn = x[:, :, self.idx_v].unsqueeze(-1)  # [seq_len, batch, 1]

        out, _ = self.rnn(x_pinn, h0)
        z = self.z_layer(out)  # [seq_len, batch, z_dim]

        # Berechne dz/dt
        dz_dt = torch.zeros_like(z)
        dz_dt[1:, :, :] = (z[1:, :, :] - z[:-1, :, :]) / self.dt

        # RNN Output: y_rnn = w1*z + w2*dz_dt + bias
        z_dzdt = torch.cat([z, dz_dt], dim=-1)  # [seq_len, batch, 2*z_dim]
        y_rnn = self.rnn_output(z_dzdt)  # [seq_len, batch, output_size]

        # Feed-Forward NN: y_nn = NN(materialremoved, F)
        materialremoved_F = torch.cat([materialremoved, F], dim=-1)  # [seq_len, batch, 2]
        y_nn = self.ffnn(materialremoved_F)  # [seq_len, batch, output_size]

        # Gesamtvorhersage
        y_pred = y_linear + y_rnn + y_nn

        return y_pred

    def train_model(self, X_train, y_train, X_val, y_val, target='curr_x', **kwargs):
        """
        Trainiert zuerst die lineare Regression, dann das RNN auf den Residuen.
        """
        self._initialize()

        # Bestimme die Indizes von a, v, F, materialremoved
        if self.idx_v is None or self.idx_a is None or self.idx_F is None:
            self._set_physics_indices(X_train, target)

        # Trainiere lineare Regression (nur einmal)
        if not self.linear_regression_trained:
            self._train_linear_regression(X_train, y_train)

        # Konvertiere zu Tensoren
        X_train_tensor = self.scaled_to_tensor(X_train)
        y_train_tensor = self.to_tensor(y_train)
        X_val_tensor = self.scaled_to_tensor(X_val)
        y_val_tensor = self.to_tensor(y_val)

        # Berechne Residuen für Training (nur lineare Regression subtrahieren)
        with torch.no_grad():
            # Lineare Vorhersagen
            if X_train_tensor.dim() == 2:
                X_train_tensor_batch = X_train_tensor.unsqueeze(1)
            else:
                X_train_tensor_batch = X_train_tensor

            a_train, v_train, F_train, _ = self._extract_physics_terms(X_train_tensor_batch)
            avF_train = torch.cat([a_train, v_train, F_train], dim=-1)
            y_linear_train = self.linear_regression(avF_train).squeeze(1)

            # Residuen
            residuals_train = y_train_tensor - y_linear_train

        # Trainiere RNN + NN auf Residuen
        print("Trainiere RNN + Feed-Forward NN auf Residuen...")
        result = super().train_model(
            X_train_tensor, residuals_train,
            X_val_tensor, y_val_tensor,  # Validierung auf echten Werten
            **kwargs
        )

        return result

    def test_model(self, X, y_target, target='curr_x', **kwargs):
        """
        Testet das Gesamtmodell (lineare Regression + RNN).
        """
        if self.idx_v is None or self.idx_a is None or self.idx_F is None:
            self._set_physics_indices(X, target)

        return super().test_model(X, y_target, **kwargs)

    def reset_hyperparameter(self, **kwargs):
        """Setzt Hyperparameter zurück und initialisiert das Modell neu."""
        if 'n_hidden_size' in kwargs:
            self.n_hidden_size = kwargs['n_hidden_size']
        if 'n_hidden_layers' in kwargs:
            self.n_hidden_layers = kwargs['n_hidden_layers']
        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']
        if 'activation' in kwargs:
            self.activation_name = kwargs['activation']
            self.activation = self.activation_map[self.activation_name]()
        if 'dropout_rate' in kwargs:
            self.dropout_rate = kwargs['dropout_rate']
        if 'z_dim' in kwargs:
            self.z_dim = kwargs['z_dim']
        if 'dt' in kwargs:
            self.dt = kwargs['dt']
        if 'axis' in kwargs:
            self.axis = kwargs['axis']

        if self.input_size is not None and any(param in kwargs for param in [
            'n_hidden_size', 'n_hidden_layers', 'activation', 'dropout_rate', 'z_dim'
        ]):
            self.linear_regression_trained = False
            self._initialize()

    def get_documentation(self):
        """Gibt eine Dokumentation des Modells zurück."""
        doc = {
            "model_type": "PhysicsInformedRNN_with_LinearRegression_and_FFNN",
            "architecture": {
                "linear_regression": "y_linear = w1*a + w2*v + w3*F (no bias, trained on |v|>1 & materialremoved>100)",
                "rnn": "learns latent z from v",
                "rnn_output": "y_rnn = w1*z + w2*dz_dt + bias",
                "ffnn": "y_nn = NN(materialremoved, F) with hidden layers",
                "final": "y_pred = y_linear + y_rnn + y_nn"
            },
            "hyperparameters": {
                "input_size": self.input_size,
                "output_size": self.output_size,
                "n_hidden_size": self.n_hidden_size,
                "n_hidden_layers": self.n_hidden_layers,
                "n_nn_hidden_size": self.n_nn_hidden_size,
                "n_nn_hidden_layers": self.n_nn_hidden_layers,
                "activation": self.activation_name,
                "learning_rate": self.learning_rate,
                "optimizer_type": self.optimizer_type,
                "dropout_rate": self.dropout_rate,
                "z_dim": self.z_dim,
                "dt": self.dt,
                "axis": self.axis,
            },
            "device": str(self.device),
            "linear_regression_trained": self.linear_regression_trained,
        }
        return doc

    @staticmethod
    def get_reference_model(input_size=None):
        """
        Returns a reference neural network model with default parameters.
        """
        return PiRNN(learning_rate=1, n_hidden_size=71, n_hidden_layers=1,
                     activation='ELU', optimizer_type='quasi_newton')

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 1  # Bei RF mit festem random state nicht sinvoll

    dataSet = hdata.DataClass_ST_Plate_Notch

    dataclass = copy.copy(dataSet)
    model =PiRNN(learning_rate= 0.1, n_hidden_size= 71, n_hidden_layers= 1,
                      activation= 'ELU', optimizer_type= 'quasi_newton')

    # Run the experiment
    hexp.run_experiment([dataclass], models=[model],
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['prediction_overview', 'model_heatmap'], experiment_name='PiRNN')