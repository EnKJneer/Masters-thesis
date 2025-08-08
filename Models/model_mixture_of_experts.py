import numpy as np
import optuna
import pandas as pd
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod

from numpy.f2py.auxfuncs import throw_error

import Models.model_base as mb
import Models.model_neural_net as mnn

# Abstrakte Klasse für die Vorverarbeitung
class AbstractPreprocessing(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

# Abstrakte Klasse für die Datenrepräsentation
class AbstractRepresentation(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

# Abstrakte Klasse für das Routing
class AbstractRouting(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

# Abstrakte Klasse für das Gating
class AbstractGating(nn.Module, ABC):
    @abstractmethod
    def forward(self, x, topk_idx, weights):
        pass

# Abstrakte Klasse für das gesamte Modell
class AbstractMoEModel(mb.BaseTorchModel, ABC):
    def __init__(self, *args, **kwargs):
        penalty_balance = kwargs.pop('penalty_balance', None)  # Remove 'penalty_balance' from kwargs

        super().__init__(*args, **kwargs)

        if penalty_balance is None:
            self.penalty_balance = 0.01
        else:
            self.penalty_balance = penalty_balance

        self.x_repr_log = []
        self.active_experts_log = []

    def _initialize(self):
        for expert in self.experts:
            expert._initialize()
            expert.to(self.device)

    def forward(self, x):
        x_pre = self.preprocessor(x)
        x_repr = self.representation(x_pre)
        weights, topk_idx = self.router(x_repr)
        # Log x_repr and active experts
        self.log_x_repr_and_active_experts(x_repr, topk_idx)

        return self.gating(x_pre, topk_idx, weights)

    def log_x_repr_and_active_experts(self, x_repr, topk_idx):
        # Convert tensors to numpy arrays for logging
        x_repr_np = x_repr.detach().cpu().numpy()
        active_experts_np = topk_idx.detach().cpu().numpy()

        self.x_repr_log.append(x_repr_np)
        self.active_experts_log.append(active_experts_np)

    def plot_active_experts(self):
        if not self.x_repr_log or not self.active_experts_log:
            print("No data to plot. Run the model first.")
            return

        # Plot x_repr over time
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        for i, x_repr in enumerate(self.x_repr_log):
            plt.plot(x_repr, label=f'Step {i+1}')
        plt.title('Temporal Evolution of x_repr')
        plt.xlabel('Time Step')
        plt.ylabel('x_repr Value')
        plt.legend()

        # Plot active experts over time
        plt.subplot(1, 2, 2)
        for i, active_experts in enumerate(self.active_experts_log):
            plt.plot(active_experts, 'o', label=f'Step {i+1}')
        plt.title('Active Experts Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Expert ID')
        plt.yticks(range(self.n_experts))
        plt.legend()

        plt.tight_layout()
        plt.show()

    def clear_active_experts_log(self):
        self.x_repr_log = []
        self.active_experts_log = []

    def criterion(self, y_target, y_pred, router_outputs=None):
        mse = F.mse_loss(y_pred.squeeze(), y_target.squeeze())
        if router_outputs is not None:
            weights, topk_idx = router_outputs
            B, k = weights.shape
            load = torch.zeros(self.n_experts, device=weights.device)
            for i in range(k):
                expert_ids = topk_idx[:, i]
                weight_i = weights[:, i]
                for expert in range(self.n_experts):
                    mask = (expert_ids == expert)
                    if mask.any():
                        load[expert] += weight_i[mask].sum()
            load = load / load.sum()
            load_balance_loss = self.n_experts * torch.sum(load * torch.log(load + 1e-9))
            return mse + self.penalty_balance * load_balance_loss
        return mse

    def get_documentation(self):
        return {"hyperparameters": {
            "input_size": self.input_size,
            "embed_dim": self.embed_dim,
            "n_experts": self.n_experts,
            "learning_rate": self.learning_rate,
        }}

# Konkrete Implementierung der Vorverarbeitung
class TransformerPreprocessing(AbstractPreprocessing):
    def __init__(self, input_dim, embed_dim, n_heads=5, n_layers=1):
        super().__init__()
        self.embed = ResidualMLP(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return x
class IdentityPreprocessing(AbstractPreprocessing):
    def forward(self, x):
        return x

# Konkrete Implementierung der Datenrepräsentation
class IdentityRepresentation(AbstractRepresentation):
    def forward(self, x):
        return x
class SignRepresentation(nn.Module):
    def forward(self, x):

        acceleration, velocity, force, MRR = self.get_input_vector_from_tensor(x)
        x = velocity[1]
        # Define a small value to determine the "approximately zero" range
        epsilon = 1e-3
        # Create a mask for values around zero
        zero_mask = torch.abs(x) < epsilon
        # Create a mask for positive values
        positive_mask = x > epsilon
        # Create a mask for negative values
        negative_mask = x < -epsilon

        # Apply the masks to create the sign representation
        x[zero_mask] = 0
        x[positive_mask] = 1
        x[negative_mask] = -1

        return x.unsqueeze(-1)

    def get_input_vector_from_tensor(self, input_vector):
        if isinstance(input_vector, list):
            acceleration, velocity, force, MRR = input_vector  # shape: [4, T] per axis
        elif isinstance(input_vector, torch.Tensor):
            if input_vector.shape[1] == 13:
                input_vector = input_vector.T
                acceleration = [input_vector[0, :], input_vector[1, :], input_vector[2, :], input_vector[3, :]]
                force = [input_vector[4, :], input_vector[5, :], input_vector[6, :], input_vector[7, :]]
                MRR = input_vector[8, :]
                velocity = [input_vector[9, :], input_vector[10, :], input_vector[11, :], input_vector[12, :]]
            elif input_vector.shape[1] == 4:
                acceleration, velocity, force, MRR = input_vector.T
            else:
                raise ValueError(f'input_vector shape {input_vector.shape}, wrong shape')
        else:
            raise ValueError(f'input_vector is of type {type(input_vector)} but should be of type list or torch.Tensor')

        return acceleration, velocity, force, MRR
class AbsRepresentation(nn.Module):
    def forward(self, x):
        acceleration, velocity, force, MRR = self.get_input_vector_from_tensor(x)
        x = velocity[1]

        # Define a small value to determine the "approximately zero" range
        epsilon = 1e-3

        # Create a mask for values around zero
        zero_mask = torch.abs(x) < epsilon

        # Apply the mask to create the absolute representation
        x[~zero_mask] = 1
        x[zero_mask] = 2  # Set values approximately zero to 1

        return x.unsqueeze(-1)

    def get_input_vector_from_tensor(self, input_vector):
        if isinstance(input_vector, list):
            acceleration, velocity, force, MRR = input_vector  # shape: [4, T] per axis
        elif isinstance(input_vector, torch.Tensor):
            if input_vector.shape[1] == 13:
                input_vector = input_vector.T
                acceleration = [input_vector[0, :], input_vector[1, :], input_vector[2, :], input_vector[3, :]]
                force = [input_vector[4, :], input_vector[5, :], input_vector[6, :], input_vector[7, :]]
                MRR = input_vector[8, :]
                velocity = [input_vector[9, :], input_vector[10, :], input_vector[11, :], input_vector[12, :]]
            elif input_vector.shape[1] == 4:
                acceleration, velocity, force, MRR = input_vector.T
            else:
                raise ValueError(f'input_vector shape {input_vector.shape}, wrong shape')
        else:
            raise ValueError(f'input_vector is of type {type(input_vector)} but should be of type list or torch.Tensor')

        return acceleration, velocity, force, MRR
# Konkrete Implementierung des Routings
class MoiraiGating(AbstractRouting):
    def __init__(self, input_dim, n_experts, k=2):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.linear = nn.Linear(input_dim, n_experts)
    # ToDo: Initialising ist Problematisch, Besser verteilung basiernd auf Gating auswahl -> Siehe FixedMoiraiGating
    def forward(self, x):
        logits = self.linear(x)
        topk_vals, topk_indices = torch.topk(logits, self.k, dim=-1)
        gate_probs = F.softmax(topk_vals, dim=-1)
        return gate_probs, topk_indices

# Explizites deterministisches Routing
class DeterministicRouting(AbstractRouting):
    def __init__(self, input_dim, n_experts, k=1):
        super().__init__()
        self.k = k
        self.n_experts = n_experts

    def forward(self, x):
        """
        Explizite Zuordnung:
        x_repr = 1.0 -> Experte 0
        x_repr = 2.0 -> Experte 1
        """
        B = x.shape[0]
        x_flat = x.squeeze(-1) if x.dim() > 1 else x

        # Deterministisches Routing
        expert_ids = torch.zeros(B, dtype=torch.long, device=x.device)

        # Exakte Zuordnung
        mask_1 = (x_flat == 1.0)  # x_repr = 1
        mask_2 = (x_flat == 2.0)  # x_repr = 2

        expert_ids[mask_1] = 0  # Experte 0 für Wert 1
        expert_ids[mask_2] = 1  # Experte 1 für Wert 2

        # Falls weitere Werte existieren, verwende Modulo
        other_mask = ~(mask_1 | mask_2)
        if other_mask.any():
            expert_ids[other_mask] = (x_flat[other_mask].long() % self.n_experts)

        topk_indices = expert_ids.unsqueeze(1)  # (B, 1)
        weights = torch.ones(B, 1, device=x.device)

        return weights, topk_indices
# LÖSUNG 2: Korrigierte lernbare Initialisierung
class FixedMoiraiGating(AbstractRouting):
    def __init__(self, input_dim, n_experts, k=2):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.linear = nn.Linear(input_dim, n_experts)

        # KRITISCH: Bessere Initialisierung für 2 diskrete Werte (1.0 und 2.0)
        with torch.no_grad():
            if input_dim == 1 and n_experts >= 2:
                # Experte 0: Soll x_repr = 1.0 bevorzugen
                # logit_0 = w_0 * x + b_0
                # Für x=1: logit_0 = w_0 + b_0 = +2 (hoch)
                # Für x=2: logit_0 = 2*w_0 + b_0 = -2 (niedrig)
                # => w_0 = -4, b_0 = +6
                self.linear.weight[0, 0] = -4.0
                self.linear.bias[0] = 6.0

                # Experte 1: Soll x_repr = 2.0 bevorzugen
                # Für x=1: logit_1 = w_1 + b_1 = -2 (niedrig)
                # Für x=2: logit_1 = 2*w_1 + b_1 = +2 (hoch)
                # => w_1 = +4, b_1 = -6
                self.linear.weight[1, 0] = 4.0
                self.linear.bias[1] = -6.0

                # Weitere Experten: Mache sie unwahrscheinlich
                for i in range(2, n_experts):
                    self.linear.weight[i, 0] = 0.0
                    self.linear.bias[i] = -10.0

    def forward(self, x):
        logits = self.linear(x)
        topk_vals, topk_indices = torch.topk(logits, self.k, dim=-1)
        gate_probs = F.softmax(topk_vals, dim=-1)
        return gate_probs, topk_indices

# Konkrete Implementierung des Gatings
class GatingNetwork(AbstractGating):
    def __init__(self, expert_list, k):
        super().__init__()
        self.experts = nn.ModuleList(expert_list)
        self.k = k

    def forward(self, x, topk_idx, weights):
        B, D = x.shape
        out = torch.zeros(B, self.experts[0].output_size).to(x.device)
        for i in range(self.k):
            idx_i = topk_idx[:, i]
            for expert_id in range(len(self.experts)):
                sel = (idx_i == expert_id)
                if sel.any():
                    out_i = self.experts[expert_id](x[sel])
                    out[sel] += weights[sel, i].unsqueeze(1) * out_i
        return out

# Konkrete Implementierung des gesamten Modells
class ModularMoiraiMoE(AbstractMoEModel):
    def __init__(self, input_size=13, output_size=1, embed_dim=13, n_experts=8, k=2,
                 n_heads=13, expert_hidden_dim=13, learning_rate=0.001, name="ModularMoiraiMoE", optimizer_type='adam'):
        super(ModularMoiraiMoE, self).__init__(input_size=input_size, output_size=output_size, name=name,
                                  learning_rate=learning_rate, optimizer_type=optimizer_type)
        self.embed_dim = embed_dim
        self.k = k
        self.n_experts = n_experts
        self.input_size = input_size
        self.preprocessor = TransformerPreprocessing(input_size, embed_dim, n_heads)
        self.representation = IdentityRepresentation()
        self.router = MoiraiGating(embed_dim, n_experts, k)
        self.experts = [mnn.Net(embed_dim, output_size, n_hidden_size=expert_hidden_dim) for _ in range(n_experts)]
        self.gating = GatingNetwork(self.experts, k)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

class NaiveMoE(AbstractMoEModel):
    def __init__(self, input_size=13, output_size=1, embed_dim=1, n_experts=4, k=1,
                 expert_hidden_dim=13, learning_rate=0.001, name="NaiveMoE", optimizer_type='adam',
                 penalty_balance = 10):
        super(NaiveMoE, self).__init__(input_size=input_size, output_size=output_size, name=name,
                                  learning_rate=learning_rate, optimizer_type=optimizer_type, penalty_balance=penalty_balance)
        self.embed_dim = embed_dim
        self.k = k
        self.n_experts = n_experts

        self.preprocessor = IdentityPreprocessing()
        self.representation = AbsRepresentation()
        self.router = MoiraiGating(embed_dim, n_experts, k)
        self.experts = [mnn.Net(input_size, output_size, n_hidden_size=expert_hidden_dim) for _ in range(n_experts)]
        self.gating = GatingNetwork(self.experts, k)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

class MoEGatingBased(mb.BaseTorchModel):
    def __init__(self, input_size=None, output_dim=1, n_experts=10, learning_rate=0.001, name="MoEGatingBased", optimizer_type='adam'):
        super(MoEGatingBased, self).__init__(input_size=input_size, output_size=output_dim, name=name, learning_rate=learning_rate, optimizer_type=optimizer_type)
        self.experts = nn.ModuleList([mnn.Net(input_size, output_dim) for _ in range(n_experts)])
        self.gating_network = GatingNetwork(input_size, n_experts)
        self.name = name
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        # Get the gating probabilities
        gating_probs = self.gating_network(x)

        # Get the outputs from each expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # Weight the expert outputs by the gating probabilities
        weighted_outputs = expert_outputs * gating_probs.unsqueeze(-1)

        # Sum the weighted outputs to get the final output
        output = weighted_outputs.sum(dim=1)

        return output

    def _initialize(self):
        self.scaler = None
        """Initialize the layers of the neural network."""
        self.gating_network.input_size = self.input_size
        self.gating_network.output_size = len(self.experts)
        self.gating_network.device = self.device
        self.gating_network._initialize()
        for expert in self.experts:
            expert.input_size = self.input_size
            expert.device = self.device
            expert._initialize()

    def get_documentation(self):
        documentation = {"hyperparameters": {
            "learning_rate": self.learning_rate,
            "n_experts": len(self.experts),
        }}
        return documentation

class ResidualMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.activation = nn.GELU()

    def forward(self, x):
        return x + self.fc2(self.activation(self.fc1(x)))

class MoiraiMoEBlock(nn.Module):
    def __init__(self, input_dim, n_experts, expert_hidden_dim, k=2):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.gating = MoiraiGating(input_dim, n_experts, k)
        self.experts = nn.ModuleList([
            mnn.Net(input_size=input_dim, output_size=input_dim, n_hidden_size=expert_hidden_dim)
            for _ in range(n_experts)
        ])

    def forward(self, x):
        """
        Parameters:
        x : Tensor of shape (B, D) – batch of input tokens

        Returns:
        Tensor of shape (B, D) – aggregated output from selected experts
        """
        B, D = x.shape
        gate_probs, topk_indices = self.gating(x)  # (B, k), (B, k)

        # Collect expert outputs for all k selected experts
        all_outputs = torch.zeros(B, self.k, D).to(x.device)  # Output tensor

        for i in range(self.k):
            expert_ids = topk_indices[:, i]  # (B,)
            x_i = x  # same input for all

            # Expertenausgabe vorbereiten
            outputs_i = torch.zeros(B, D).to(x.device)

            # Für jeden Experten alle Samples auswählen, die ihn verwenden
            for expert_id in range(self.n_experts):
                mask = (expert_ids == expert_id)
                if mask.sum() == 0:
                    continue
                selected_x = x_i[mask]
                out = self.experts[expert_id](selected_x)
                outputs_i[mask] = out

            all_outputs[:, i, :] = outputs_i  # Speichere Ergebnisse für Experten i

        # Aggregiere alle k Expertenausgaben mit ihren Gewichten
        gate_probs = gate_probs.unsqueeze(-1)  # (B, k, 1)
        final_output = (all_outputs * gate_probs).sum(dim=1)  # (B, D)

        return final_output

class MoiraiMoEModel(mb.BaseTorchModel):
    """
    Based on: Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts, Liu 2024 https://arxiv.org/abs/2410.10469
    Simplified and adapted for this setting
    """
    def __init__(self, input_size=None, output_size=1, embed_dim=65, n_experts=8, expert_hidden_dim=65,
                 learning_rate=0.001, name="MoiraiMoE", optimizer_type='adam'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.embed_dim = embed_dim # ToDo: Flexibler gestalten
        self.n_experts = n_experts
        self.expert_hidden_dim = expert_hidden_dim
        self.learning_rate = learning_rate
        self.name = name
        self.optimizer_type = optimizer_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        if self.input_size is not None:
            self._initialize()

    def criterion(self, y_target, y_pred, router_outputs=None):
        mse = F.mse_loss(y_pred.squeeze(), y_target.squeeze())
        if router_outputs is not None:
            weights, topk_idx = router_outputs  # (B, k), (B, k)
            B, k = weights.shape
            load = torch.zeros(self.n_experts, device=weights.device)
            for i in range(k):
                expert_ids = topk_idx[:, i]
                weight_i = weights[:, i]
                for expert in range(self.n_experts):
                    mask = (expert_ids == expert)
                    if mask.any():
                        load[expert] += weight_i[mask].sum()
            load = load / load.sum()
            load_balance_loss = self.n_experts * torch.sum(load * torch.log(load + 1e-9))
            return mse + 0.01 * load_balance_loss
        return mse

    def _initialize(self):
        self.tokenizer = ResidualMLP(self.input_size, self.embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=5, batch_first=True) # ToDo: embed_dim must be divisible by num_heads -> Flexibler
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.moe_block = MoiraiMoEBlock(self.embed_dim, self.n_experts, self.expert_hidden_dim, k=2)
        self.output_layer = nn.Linear(self.embed_dim, self.output_size)
        self.to(self.device)

    def forward(self, x):
        if self.input_size is None:
            self.input_size = x.shape[1]
            self._initialize()

        x = self.tokenizer(x)  # (B, D)
        x = x.unsqueeze(1)  # (B, 1, D)
        x = self.transformer(x)  # (B, 1, D)
        x = x.squeeze(1)  # (B, D)
        x = self.moe_block(x)  # (B, D)
        x = self.output_layer(x)  # (B, output_size)
        return x

    def get_documentation(self):
        return {"hyperparameters": {
            "input_size": self.input_size,
            "embed_dim": self.embed_dim,
            "n_experts": self.n_experts,
            "expert_hidden_dim": self.expert_hidden_dim,
            "learning_rate": self.learning_rate,
        }}

# LÖSUNG 4: Aktualisierte NaiveMoE
class TrulyNaiveMoE(AbstractMoEModel):
    def __init__(self, input_size=13, output_size=1, embed_dim=1, n_experts=4, k=1,
                 expert_hidden_dim=13, learning_rate=0.001, name="TrulyNaiveMoE",
                 optimizer_type='adam', penalty_balance=0.0, routing_strategy='deterministic'):

        super(TrulyNaiveMoE, self).__init__(
            input_size=input_size, output_size=output_size, name=name,
            learning_rate=learning_rate, optimizer_type=optimizer_type,
            penalty_balance=penalty_balance
        )

        self.embed_dim = embed_dim
        self.k = k
        self.n_experts = n_experts

        self.preprocessor = IdentityPreprocessing()
        self.representation = AbsRepresentation()

        # Wähle Routing-Strategie
        if routing_strategy == 'deterministic':
            self.router = DeterministicRouting(embed_dim, n_experts, k)
        elif routing_strategy == 'fixed_init':
            self.router = FixedMoiraiGating(embed_dim, n_experts, k)
        else:
            self.router = MoiraiGating(embed_dim, n_experts, k)

        self.experts = [mnn.Net(input_size, output_size, n_hidden_size=expert_hidden_dim)
                        for _ in range(n_experts)]
        self.gating = GatingNetwork(self.experts, k)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

# Memory-Enhanced Routing für das MoE
class MemoryEnhancedRouting(AbstractRouting):
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
class MemoryEnhancedGating(AbstractGating):
    def __init__(self, expert_list, k, input_size):
        super().__init__()
        self.experts = nn.ModuleList(expert_list)
        self.k = k
        self.input_size = input_size

        # Speicher für die letzte Vorhersage bei Velocity ≠ 0
        self.last_nonzero_prediction = None

    def forward(self, x, topk_idx, weights):
        # x: [T, D]
        # topk_idx: [T, k] → sagt, welcher Experte aktiv ist (0 oder 1)
        # weights: [T, k]  → Gewichtungen (k=1 für dein Fall)

        device = x.device
        T, D = x.shape
        out_dim = self.experts[0].output_size

        # Maske: Expert 0 überall dort, wo mind. einer topk_idx==0
        expert_0_mask = (topk_idx == 0).any(dim=1)  # [T]
        expert_1_mask = (topk_idx == 1).any(dim=1)  # [T]

        # --- Expert 0 ganz normal ---
        out_0 = torch.zeros(T, out_dim, device=device)
        if expert_0_mask.any():
            out_0[expert_0_mask] = self.experts[0](x[expert_0_mask])

        # --- Fülle zuletzt gültigen out_0 ---
        last_valid_out_0 = self.last_valid_fill(out_0, expert_0_mask)

        # --- Expert 1 bekommt erweiterten Input ---
        x_exp1 = x[expert_1_mask]
        y_prev = last_valid_out_0[expert_1_mask]
        x_exp1_enhanced = torch.cat([x_exp1, y_prev], dim=1)

        out_1 = torch.zeros(T, out_dim, device=device)
        if expert_1_mask.any():
            out_1[expert_1_mask] = self.experts[1](x_exp1_enhanced)

        # --- Final kombinieren mit weights ---
        # Hinweis: Falls du k=1 hast, kannst du einfach:
        w = weights[:, 0].unsqueeze(1)  # [T, 1]
        out = w * out_0 + (1 - w) * out_1  # gewichtete Summe

        return out

    def last_valid_fill(self, data: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # data: [T, D]
        # valid_mask: [T] (bool)

        T, D = data.shape
        device = data.device

        # Indizes der Zeitachse
        time = torch.arange(T, device=device)

        # Ersetze ungültige durch -1, gültige durch ihre Zeitposition
        valid_time = torch.where(valid_mask, time, torch.full((T,), -1, device=device))

        # Hole für jeden Zeitschritt das zuletzt gültige Zeit-Index
        last_valid_time = valid_time.clone()
        last_valid_time[last_valid_time == -1] = 0  # Vorbereitung für cummax
        cummax_time, _ = last_valid_time.unsqueeze(1).expand(-1, D).cummax(dim=0)

        # Baue Tensor [T, D] mit zuletzt gültigen Werten
        filled = data[cummax_time, torch.arange(D).unsqueeze(0).expand(T, -1)]

        return filled

    def reset_memory(self):
        """Reset der gespeicherten Vorhersage für neue Sequenzen"""
        self.last_nonzero_prediction = None

# Lineare Regression als Experte
class LinearRegressionExpert(nn.Module):
    def __init__(self, input_size, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.device = None

    def forward(self, x):
        return self.linear(x)

    def _initialize(self):
        """Kompatibilität mit BaseNetModel Interface"""
        if self.device is not None:
            self.to(self.device)

# Memory-Enhanced MoE Model
class MemoryEnhancedMoE(AbstractMoEModel):
    def __init__(self, input_size=13, output_size=1, embed_dim=1, n_experts=2, k=1,
                 expert_hidden_dim=13, learning_rate=0.001, name="MemoryEnhancedMoE",
                 optimizer_type='adam', penalty_balance=0.0, axis=1):
        super(MemoryEnhancedMoE, self).__init__(
            input_size=input_size, output_size=output_size, name=name,
            learning_rate=learning_rate, optimizer_type=optimizer_type,
            penalty_balance=penalty_balance
        )

        self.embed_dim = embed_dim
        self.k = k
        self.n_experts = n_experts
        self.axis = axis  # Welche Velocity-Achse verwendet werden soll

        # Komponenten
        self.preprocessor = IdentityPreprocessing()
        self.representation = AbsRepresentation()  # Verwendet bereits axis=1 intern
        self.router = MemoryEnhancedRouting(embed_dim, n_experts, k)

        # Experte 0: Lineare Regression für normalen Input (Velocity ≠ 0)
        # Experte 1: Lineare Regression für erweiterten Input (Original + letzte Vorhersage) für Velocity ≈ 0
        self.experts = [
            mnn.Net(input_size, output_size),  # Experte 0
            mnn.Net(input_size + output_size, output_size)  # Experte 1
        ]

        self.gating = MemoryEnhancedGating(self.experts, k, input_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

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
            },
            "description": "Memory-Enhanced MoE with velocity-based routing and prediction memory"
        }
    @staticmethod
    def reset_hyperparameter():
        throw_error('Not implemented yet')