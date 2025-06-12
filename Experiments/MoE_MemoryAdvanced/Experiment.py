import os

import numpy as np
import pandas as pd
import torch
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn
import Models.model_physical as mphys
import Models.model_random_forest as mrf
import Models.model_mixture_of_experts as mmix


# Memory-Enhanced Routing für das MoE
class MemoryEnhancedRouting(mmix.AbstractRouting):
    def __init__(self, input_dim, n_experts=2, k=1):
        super().__init__()
        self.k = k
        self.n_experts = n_experts

    def forward(self, x):
        """
        Deterministisches Routing basierend auf AbsRepresentation:
        x = 1.0 -> Experte 0 (Velocity ≠ 0)
        x = 2.0 -> Experte 1 (Velocity ≈ 0)
        """
        B = x.shape[0]
        x_flat = x.squeeze(-1) if x.dim() > 1 else x

        # Deterministisches Routing
        expert_ids = torch.zeros(B, dtype=torch.long, device=x.device)

        # Zuordnung basierend auf AbsRepresentation
        mask_velocity_nonzero = (x_flat == 1.0)  # Velocity ≠ 0 -> Experte 0
        mask_velocity_zero = (x_flat == 2.0)  # Velocity ≈ 0 -> Experte 1

        expert_ids[mask_velocity_nonzero] = 0  # Experte 0 für Velocity ≠ 0
        expert_ids[mask_velocity_zero] = 1  # Experte 1 für Velocity ≈ 0

        topk_indices = expert_ids.unsqueeze(1)  # (B, 1)
        weights = torch.ones(B, 1, device=x.device)

        return weights, topk_indices

# Memory-Enhanced Gating Network
class MemoryEnhancedGating(mmix.AbstractGating):
    def __init__(self, expert_list, k, input_size):
        super().__init__()
        self.experts = nn.ModuleList(expert_list)
        self.k = k
        self.input_size = input_size

        # Speicher für die letzte Vorhersage bei Velocity ≠ 0
        self.last_nonzero_prediction = None

    def forward(self, x, topk_idx, weights):
        B, D = x.shape
        out = torch.zeros(B, self.experts[0].output_size).to(x.device)

        for i in range(self.k):
            idx_i = topk_idx[:, i]

            # Experte 0: Velocity ≠ 0 (normaler Input)
            sel_expert_0 = (idx_i == 0)
            if sel_expert_0.any():
                x_expert_0 = x[sel_expert_0]
                out_0 = self.experts[0](x_expert_0)
                out[sel_expert_0] += weights[sel_expert_0, i].unsqueeze(1) * out_0

                # Speichere die Vorhersage für späteren Gebrauch
                # Nehme die letzte Vorhersage aus dem Batch
                self.last_nonzero_prediction = out_0[-1:].detach().clone()

            # Experte 1: Velocity ≈ 0 (erweiteter Input mit letzter Vorhersage)
            sel_expert_1 = (idx_i == 1)
            if sel_expert_1.any():
                x_expert_1 = x[sel_expert_1]

                # Erweitere Input um die letzte bekannte Vorhersage
                if self.last_nonzero_prediction is None:
                    # Fallback am Anfang: y = 0
                    last_pred = torch.zeros(1, self.experts[1].output_size).to(x.device)
                else:
                    last_pred = self.last_nonzero_prediction

                # Erweitere jeden Sample im Batch um die letzte Vorhersage
                batch_size_expert_1 = x_expert_1.shape[0]
                last_pred_expanded = last_pred.expand(batch_size_expert_1, -1)

                # Konkateniere Original-Input mit letzter Vorhersage
                x_expert_1_enhanced = torch.cat([x_expert_1, last_pred_expanded], dim=1)

                out_1 = self.experts[1](x_expert_1_enhanced)
                out[sel_expert_1] += weights[sel_expert_1, i].unsqueeze(1) * out_1

        # Einfache Addition (kein gewichtetes Gating zwischen Experten)
        return out

    def reset_memory(self):
        """Reset der gespeicherten Vorhersage für neue Sequenzen"""
        self.last_nonzero_prediction = None

# Random Forest als Experte
class RandomForestExpert(nn.Module):
    def __init__(self, input_size, output_size=1, n_estimators=10, max_features=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        super().__init__()
        from sklearn.ensemble import RandomForestRegressor

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        self.input_size = input_size
        self.output_size = output_size
        self.device = "cpu"  # Random Forest läuft immer auf CPU
        self.is_fitted = False

    def forward(self, x):
        # Konvertiere PyTorch Tensor zu NumPy für sklearn
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x

        if not self.is_fitted:
            # Fallback: Gib Nullen zurück wenn noch nicht trainiert
            batch_size = x_np.shape[0]
            return torch.zeros(batch_size, self.output_size)

        # Vorhersage
        pred = self.model.predict(x_np)

        # Konvertiere zurück zu PyTorch Tensor
        if isinstance(x, torch.Tensor):
            return torch.tensor(pred, dtype=x.dtype, device=x.device).reshape(-1, self.output_size)
        else:
            return pred.reshape(-1, self.output_size)

    def fit(self, X, y):
        """Trainiere den Random Forest"""
        # Konvertiere zu NumPy falls nötig
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy().ravel()

        self.model.fit(X, y)
        self.is_fitted = True

    def _initialize(self):
        """Kompatibilität mit BaseNetModel Interface"""
        pass  # Random Forest braucht keine spezielle Initialisierung

# Memory-Enhanced MoE Model
class MemoryEnhancedMoE(mmix.AbstractMoEModel):
    def __init__(self, input_size=13, output_size=1, embed_dim=1, n_experts=2, k=1,
                 expert_hidden_dim=13, learning_rate=0.001, name="MemoryEnhancedMoE",
                 optimizer_type='adam', penalty_balance=0.0, axis=1,
                 rf_n_estimators=10, rf_max_features=None, rf_min_samples_split=2,
                 rf_min_samples_leaf=1, rf_random_state=None):

        super(MemoryEnhancedMoE, self).__init__(
            input_size=input_size, output_size=output_size, name=name,
            learning_rate=learning_rate, optimizer_type=optimizer_type,
            penalty_balance=penalty_balance
        )

        self.embed_dim = embed_dim
        self.k = k
        self.n_experts = n_experts
        self.axis = axis  # Welche Velocity-Achse verwendet werden soll

        # Random Forest Parameter
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_features = rf_max_features
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_random_state = rf_random_state

        # Komponenten
        self.preprocessor = mmix.IdentityPreprocessing()
        self.representation = mmix.AbsRepresentation()  # Verwendet bereits axis=1 intern
        self.router = MemoryEnhancedRouting(embed_dim, n_experts, k)

        # Experte 0: Random Forest für normalen Input (Velocity ≠ 0)
        # Experte 1: Random Forest für erweiterten Input (Original + letzte Vorhersage) für Velocity ≈ 0
        self.experts = [
            RandomForestExpert(input_size, output_size,
                               n_estimators=rf_n_estimators,
                               max_features=rf_max_features,
                               min_samples_split=rf_min_samples_split,
                               min_samples_leaf=rf_min_samples_leaf,
                               random_state=rf_random_state),  # Experte 0
            RandomForestExpert(input_size + output_size, output_size,
                               n_estimators=rf_n_estimators,
                               max_features=rf_max_features,
                               min_samples_split=rf_min_samples_split,
                               min_samples_leaf=rf_min_samples_leaf,
                               random_state=rf_random_state)  # Experte 1
        ]

        self.gating = MemoryEnhancedGating(self.experts, k, input_size)

        self.device = torch.device('cpu')  # Random Forest läuft auf CPU
        # Nur die Routing-Komponenten auf GPU falls verfügbar
        if torch.cuda.is_available():
            self.preprocessor.to('cuda')
            self.representation.to('cuda')
            self.router.to('cuda')

    def fit_experts(self, X, y, routing_labels):
        """
        Trainiere die Random Forest Experten mit den entsprechenden Daten

        Parameters:
        -----------
        X : torch.Tensor or numpy.ndarray
            Input data
        y : torch.Tensor or numpy.ndarray
            Target data
        routing_labels : torch.Tensor or numpy.ndarray
            Labels für das Routing (0 für Experte 0, 1 für Experte 1)
        """
        # Konvertiere zu NumPy falls nötig
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(routing_labels, torch.Tensor):
            routing_labels = routing_labels.detach().cpu().numpy()

        # Trainiere Experte 0 (Velocity ≠ 0)
        mask_expert_0 = (routing_labels == 0)
        if mask_expert_0.sum() > 0:
            X_expert_0 = X[mask_expert_0]
            y_expert_0 = y[mask_expert_0]
            self.experts[0].fit(X_expert_0, y_expert_0)

        # Trainiere Experte 1 (Velocity ≈ 0)
        mask_expert_1 = (routing_labels == 1)
        if mask_expert_1.sum() > 0:
            X_expert_1 = X[mask_expert_1]
            y_expert_1 = y[mask_expert_1]

            # Für Experte 1: Erweitere Input um letzte Vorhersage
            # Hier vereinfacht: Nutze den Durchschnitt der y-Werte als "letzte Vorhersage"
            if hasattr(self, '_last_prediction_fallback'):
                last_pred = self._last_prediction_fallback
            else:
                last_pred = y_expert_1.mean() if len(y_expert_1) > 0 else 0.0

            last_pred_expanded = np.full((X_expert_1.shape[0], 1), last_pred)
            X_expert_1_enhanced = np.concatenate([X_expert_1, last_pred_expanded], axis=1)

            self.experts[1].fit(X_expert_1_enhanced, y_expert_1)

    def generate_routing_labels(self, X):
        """
        Generiere Routing-Labels basierend auf der Velocity

        Parameters:
        -----------
        X : torch.Tensor or numpy.ndarray
            Input data

        Returns:
        --------
        routing_labels : numpy.ndarray
            Labels für das Routing (0 für Experte 0, 1 für Experte 1)
        """
        # Durchlaufe die Preprocessing-Pipeline
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        x_pre = self.preprocessor(X)
        x_repr = self.representation(x_pre)

        # Konvertiere zu NumPy
        x_repr_np = x_repr.detach().cpu().numpy().squeeze()

        # Generiere Labels basierend auf AbsRepresentation
        routing_labels = np.where(x_repr_np == 1.0, 0, 1)  # 1.0 -> Experte 0, 2.0 -> Experte 1

        return routing_labels

    def reset_memory(self):
        """Reset für neue Sequenzen"""
        self.gating.reset_memory()
        self.clear_active_experts_log()

    def get_documentation(self):
        return {
            "hyperparameters": {
                "input_size": self.input_size,
                "embed_dim": self.embed_dim,
                "n_experts": self.n_experts,
                "axis": self.axis,
                "learning_rate": self.learning_rate,
                "rf_n_estimators": self.rf_n_estimators,
                "rf_max_features": self.rf_max_features,
                "rf_min_samples_split": self.rf_min_samples_split,
                "rf_min_samples_leaf": self.rf_min_samples_leaf,
            },
            "description": "Memory-Enhanced MoE with velocity-based routing and Random Forest experts"
        }

if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 2

    window_size = 1
    past_values = 0
    future_values = 0

    #Combined_Gear,Combined_KL
    dataClass_1 = hdata.Combined_PK_TrainVal
    dataClass_1.window_size = window_size
    dataClass_1.past_values = past_values
    dataClass_1.future_values = future_values

    dataSets_list = [dataClass_1]

    model = mmix.MemoryEnhancedMoE()
    models = [model]

    # Run the experiment
    hexp.run_experiment(dataSets_list, use_nn_reference=True, use_rf_reference=False, models=models, NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, window_size=window_size, past_values=past_values, future_values=future_values)