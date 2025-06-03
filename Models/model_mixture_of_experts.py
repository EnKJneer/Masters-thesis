import numpy as np
import optuna
import pandas as pd
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from numpy.f2py.auxfuncs import throw_error
from sklearn.preprocessing import QuantileTransformer

import Models.model_base as mb
import Models.model_neural_net as mnn

class GatingNetwork(mb.BaseNetModel):
    def __init__(self, input_size=None, output_size=10, n_hidden_size=10, n_hidden_layers=1, activation=nn.ReLU, learning_rate=0.001, name="Gating_Network", optimizer_type='adam'):
        super(GatingNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation()
        self.learning_rate = learning_rate
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        self.optimizer_type = optimizer_type
        # Initialize layers only if input_size is provided
        if self.input_size is not None:
            self._initialize()

    def _initialize(self):
        """Initialize the layers of the neural network."""
        self.scaler = None
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size
        self.fc1 = nn.Linear(self.input_size, self.n_hidden_size)
        self.fcs = nn.ModuleList([nn.Linear(self.n_hidden_size, self.n_hidden_size) for _ in range(self.n_hidden_layers)])
        self.fc3 = nn.Linear(self.n_hidden_size, self.output_size)
        self.to(self.device)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the neural network.

        Returns
        -------
        x : torch.Tensor
            The output tensor from the neural network.
        """
        if self.input_size is None:
            self.input_size = x.shape[1]
            if self.n_hidden_size is None:
                self.n_hidden_size = self.input_size
            self._initialize()

        x = self.activation(self.fc1(x))  # apply the activation function
        for fc in self.fcs:
            x = self.activation(fc(x))  # apply the activation function
        x = self.fc3(x)
        return  F.softmax(x, dim=-1)

    def get_documentation(self):
        documentation = {"hyperparameters": {
            "input_dim": self.input_dim,
            "n_experts": self.n_experts,
            "learning_rate": self.learning_rate,
        }}
        return documentation

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