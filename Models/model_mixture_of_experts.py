import numpy as np
import optuna
import pandas as pd
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from numpy.f2py.auxfuncs import throw_error
from sklearn.preprocessing import QuantileTransformer
from abc import ABC, abstractmethod

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
class AbstractMoEModel(mb.BaseNetModel, ABC):

    def _initialize(self):
        for expert in self.experts:
            expert._initialize()
            expert.to(self.device)

    def forward(self, x):
        x_pre = self.preprocessor(x)
        x_repr = self.representation(x_pre)
        weights, topk_idx = self.router(x_repr)
        return self.gating(x_repr, topk_idx, weights) # ToDo: x_pre

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
            return mse + 0.01 * load_balance_loss
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

# Konkrete Implementierung des Routings
class MoiraiGating(AbstractRouting):
    def __init__(self, input_dim, n_experts, k=2):
        super().__init__()
        self.k = k
        self.n_experts = n_experts
        self.linear = nn.Linear(input_dim, n_experts)

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
    def __init__(self, input_size=None, output_size=1, embed_dim=65, n_experts=8, k=2,
                 n_heads=5, expert_hidden_dim=65, learning_rate=0.001, name="ModularMoiraiMoE", optimizer_type='adam'):
        super(ModularMoiraiMoE, self).__init__(input_size=input_size, output_size=output_size, name=name,
                                  learning_rate=learning_rate, optimizer_type=optimizer_type)
        self.embed_dim = embed_dim
        self.k = k
        self.n_experts = n_experts

        self.preprocessor = TransformerPreprocessing(input_size, embed_dim, n_heads)
        self.representation = IdentityRepresentation()
        self.router = MoiraiGating(embed_dim, n_experts, k)
        self.experts = [mnn.Net(embed_dim, output_size, n_hidden_size=expert_hidden_dim) for _ in range(n_experts)]
        self.gating = GatingNetwork(self.experts, k)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

class NaiveMoE(AbstractMoEModel):
    def __init__(self, input_size=13, output_size=1, embed_dim=1, n_experts=8, k=2,
                 n_heads=5, expert_hidden_dim=13, learning_rate=0.001, name="NaiveMoE", optimizer_type='adam'):
        super(NaiveMoE, self).__init__(input_size=input_size, output_size=output_size, name=name,
                                  learning_rate=learning_rate, optimizer_type=optimizer_type)
        self.embed_dim = embed_dim
        self.k = k
        self.n_experts = n_experts

        self.preprocessor = IdentityPreprocessing()
        self.representation = SignRepresentation()
        self.router = MoiraiGating(embed_dim, n_experts, k)
        self.experts = [mnn.Net(input_size, output_size, n_hidden_size=expert_hidden_dim) for _ in range(n_experts)]
        self.gating = GatingNetwork(self.experts, k)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


class MoEGatingBased(mb.BaseNetModel):
    def __init__(self, input_size=None, output_dim=1, n_experts=10, learning_rate=0.001, name="MoEGatingBased", optimizer_type='adam'):
        super(MoEGatingBased, self).__init__()
        self.experts = nn.ModuleList([mnn.Net(input_size, output_dim) for _ in range(n_experts)])
        self.gating_network = GatingNetwork(input_size, n_experts)
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.name = name
        self.input_size = input_size
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

class MoiraiMoEModel(mb.BaseNetModel):
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
