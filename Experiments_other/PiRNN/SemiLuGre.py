import copy
import datetime
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_experiment as hexp
import Models.model_random_forest as mrf
import Models.model_neural_net as mnn
import Models.model_physical as mphy

class PiRNN(mnn.RNN):
    def __init__(self, *args, name="PiRNN", penalty_weight=1, optimizer_type='quasi_newton', learning_rate=0.1, theta_init=None,**kwargs):
        super(PiRNN, self).__init__(*args, learning_rate=learning_rate, **kwargs)
        self.name = name
        self.penalty_weight = penalty_weight
        self.optimizer_type = optimizer_type
        self.theta_init = theta_init
        self.input_head = hdata.HEADER_x
        # Neue Parameter-Matrix für alle Achsen
        if theta_init is not None:
            if theta_init.shape != (4, 6):
                raise ValueError("theta_init must have shape (4, 5)")
            self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        else:
            self.theta = nn.Parameter(torch.zeros(4, 6, dtype=torch.float32))  # Achsen: x, y, z, sp

    def get_indices_by_prefix(self, header, prefix):
        return [index for index, item in enumerate(header) if item.startswith(prefix)]

    def criterion(self, y_target, y_pred, x_input=None):
        criterion = nn.MSELoss()
        mse_loss = criterion(y_target.squeeze(), y_pred.squeeze())

        if x_input is not None and (y_pred.requires_grad or y_target.requires_grad):
            x_input = x_input.clone().detach().requires_grad_(True)

            #ToDO: andere lösung finden & CUDA ermöglichen

            # Temporär CuDNN deaktivieren, um "double backwards" zu ermöglichen
            with torch.backends.cudnn.flags(enabled=False):
                y_pred_physics = self(x_input)

            dy_dx = torch.autograd.grad(
                outputs=y_pred_physics,
                inputs=x_input,
                grad_outputs=torch.ones_like(y_pred_physics),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            axis = self.target_channel.replace('curr_', '')
            indices_a = self.get_indices_by_prefix(self.input_head, f'a_{axis}')
            indices_f = self.get_indices_by_prefix(self.input_head, f'f_{axis}')
            indices_v = self.get_indices_by_prefix(self.input_head, f'v_{axis}')
            indices_z = self.get_indices_by_prefix(self.input_head, f'z_{axis}')

            all_excluded_indices = set(indices_a + indices_f + indices_v + indices_z)
            remaining_indices = [i for i in range(dy_dx.shape[1]) if i not in all_excluded_indices]

            dy_remaining = dy_dx[:, remaining_indices]
            dy_da = dy_dx[:, indices_a]
            dy_df = dy_dx[:, indices_f]
            dy_dv = dy_dx[:, indices_v]
            dy_dz = dy_dx[:, indices_z]

            a = x_input[:, indices_a]
            f = x_input[:, indices_f]
            v = x_input[:, indices_v]
            z = x_input[:, indices_z]

            # Berechne die zeitliche Ableitung der Modellausgabe (y_pred_physics)
            y_pred_physics_squeezed = y_pred_physics.squeeze()
            dy_dt = torch.diff(y_pred_physics_squeezed, dim=0)  # Differenz berechnen

            # Ergänze die fehlenden Werte mit Nullen
            if dy_dt.dim() == 1:
                dy_dt = dy_dt.unsqueeze(1)  # Falls nötig, Dimension anpassen

            # Füge eine Null am Anfang hinzu, um die Länge zu erhalten
            dy_dt = torch.cat([torch.zeros(1, dy_dt.shape[1], device=self.device), dy_dt], dim=0)

            # Falls die ursprüngliche Länge nicht erreicht ist, füge eine zusätzliche Null am Ende hinzu
            if dy_dt.size(0) < y_pred_physics_squeezed.size(0):
                dy_dt = torch.cat([dy_dt, torch.zeros(1, dy_dt.shape[1], device=self.device)], dim=0)

            # Neuer Constraint: dy_dt hängt linear von v und y ab, aber nur wo |v| < 1
            theta_dydt_v = self.theta[:, 4]  # Parameter für den v-Term
            theta_dydt_y = self.theta[:, 5]  # Parameter für den y-Term (erweitere theta um eine Spalte)

            # Anpassung für unterschiedliche Dimensionen von v
            if v.dim() == 2:
                mask = (torch.abs(v) < 1).float()
            else:
                mask = (torch.abs(v) < 1).all(dim=2, keepdim=True).float()

            # Wende die Maske auf dy_dt, v, und y an
            masked_dy_dt = dy_dt * mask
            masked_v = v * mask
            masked_y = y_pred_physics_squeezed.unsqueeze(-1) * mask

            # Berechne den Constraint nur für die maskierten Bereiche
            if v.dim() == 2:
                constraint_dy_dt = masked_dy_dt - (masked_v * theta_dydt_v + masked_y.squeeze() * theta_dydt_y)
            else:
                constraint_dy_dt = masked_dy_dt - torch.sum(masked_v * theta_dydt_v, dim=1) - torch.sum(
                    masked_y * theta_dydt_y, dim=1)

            penalty_dy_dt = torch.mean(constraint_dy_dt ** 2)

            # --- Ursprünglicher Constraint ---
            constraint = []
            deriv = dy_df + dy_da
            influences = (
                    f * self.theta[:, 2] +
                    a * self.theta[:, 3]
            )
            constraint_i = deriv - influences
            constraint.append(constraint_i.unsqueeze(2))
            constraint = torch.cat(constraint, dim=2)
            penalty_original = torch.mean(constraint ** 2)

            # Gesamtpenalty
            penalty = penalty_original + penalty_dy_dt  # Gewichtung kann angepasst werden

            return mse_loss + self.penalty_weight * penalty

        return mse_loss

    def scaled_to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif hasattr(data, 'values'):
            data_scaled = self.scale_data(data.values)
            return torch.tensor(data_scaled, dtype=torch.float32).to(self.device)
        else:
            data_scaled = self.scale_data(data)
            return torch.tensor(data_scaled, dtype=torch.float32).to(self.device)

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        original_criterion = self.criterion

        self.input_head = [col for col in X_train.columns if "_1_current" in col]
        current_input = self.scaled_to_tensor(X_train)

        def custom_train_model(*args, **kwargs):
            nonlocal current_input
            original_train_model = super(PiRNN, self).train_model

            def patched_criterion(y_target, y_pred):
                return original_criterion(y_target, y_pred, x_input=current_input)

            self.criterion = patched_criterion
            result = original_train_model(*args, **kwargs)
            self.criterion = original_criterion
            return result

        return custom_train_model(X_train, y_train, X_val, y_val, **kwargs)

    def get_documentation(self):
        documentation = {"hyperparameters": {
            "learning_rate": self.learning_rate,
            "n_hidden_size": self.n_hidden_size,
            "n_hidden_layers": self.n_hidden_layers,
            "n_activation_function": self.activation.__class__.__name__,
            "optimizer_type": self.optimizer_type,
            "penalty_weight": self.penalty_weight,
        }}
        return documentation

if __name__ == '__main__':

    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 5

    window_size = 1
    past_values = 0
    future_values = 0

    dataSet = hdata.DataClass_ST_Plate_Notch

    dataclass2 = copy.copy(dataSet)
    dataclass2.name = 'mit z'
    dataclass2.add_sign_hold = True
    dataClasses = [dataclass2]
    for dataclass in dataClasses:
        dataclass.window_size = window_size
        dataclass.past_values = past_values
        dataclass.future_values = future_values
        dataclass.add_padding = True

    model_ref = mrf.RandomForestModel()
    model_1 = PiRNN()

    models = [model_1, model_ref]

    # Run the experiment
    hexp.run_experiment(dataClasses, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name='PiRNN_Friction')
