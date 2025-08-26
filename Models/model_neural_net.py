# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:20:38 2024

@author: Jonas Kyrion

SKRIPT DESCRIPTION:

Contains parameterizable neural network with necessary training functions
"""
import numpy as np
import optuna
import pandas as pd
import torch
import torchcde
import torch.jit
import torch.nn as nn
from numpy.f2py.auxfuncs import throw_error
from sklearn.preprocessing import QuantileTransformer

import Models.model_base as mb
import Helper.handling_data as hdata

def get_reference(input_size=None):
    """
    Get a reference neural network with specified input size.

    This function initializes and returns a neural network model with the given input size.
    The output size and hidden size are set to 1 and the input size, respectively.

    Parameters
    ----------
    input_size : int
        The number of input features for the neural network.
        Default is None. -> Input will be set at runtime.
    Returns
    -------
    Net
        An instance of the Net class configured with the specified input size.
    """
    return Net(input_size, 1, input_size)

# Defines a configurable neural network
class Net(mb.BaseTorchModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=1, activation='ReLU',
                 learning_rate=0.001, name="Neural_Net", optimizer_type='adam'):
        """
        Initializes a configurable neural network.

        Parameters
        ----------
        input_size : int, optional
            The number of input features. If None, it will be set during the first training call.
        output_size : int
            The number of output features.
        n_hidden_size : int, optional
            The number of features in each hidden layer. If None, it will be set to the input size during the first training call.
        n_hidden_layers : int
            The number of hidden layers in the network.
        activation : torch.nn.Module, optional
            The activation function to be used in the hidden layers. The default is nn.ReLU.
        learning_rate : float
            The learning rate for the optimizer.
        name : str
            The name of the model.
        """
        super(Net, self).__init__(input_size=input_size, output_size=output_size, name=name, learning_rate=learning_rate, optimizer_type=optimizer_type)

        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers

        self.activation = self.activation_map[activation]()

        # Initialize layers only if input_size is provided
        if self.input_size is not None:
            self._initialize()

    def reset_hyperparameter(self, n_hidden_size, n_hidden_layers, learning_rate, activation, optimizer_type):
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate
        self.activation = self.activation_map[activation]()
        self.optimizer_type = optimizer_type

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
        return x
    @staticmethod
    def get_reference_model(input_size=None):
        """
        Get a reference neural network with specified input size.

        This function initializes and returns a neural network model with the given input size.
        The output size and hidden size are set to 1 and the input size, respectively.

        Parameters
        ----------
        input_size : int
            The number of input features for the neural network.
            Default is None. -> Input will be set at runtime.
        Returns
        -------
        Net
            An instance of the Net class configured with the specified input size.
        """
        return Net(input_size, 1, input_size)

class RNN(mb.BaseTorchModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=1, activation='ReLU',
                 learning_rate=0.001, name="Recurrent_Neural_Net", batched_input=False, optimizer_type='adam'):
        """
        Initializes a configurable recurrent neural network.

        Parameters
        ----------
        input_size : int, optional
            The number of input features. If None, it will be set during the first training call.
        output_size : int
            The number of output features.
        n_hidden_size : int, optional
            The number of features in each hidden layer. If None, it will be set to the input size during the first training call.
        n_hidden_layers : int
            The number of hidden layers in the network.
        activation : torch.nn.Module, optional
            The activation function to be used in the hidden layers. The default is nn.ReLU.
        name: str
            The name of the model.
        batched_input : bool
            Is the input batched or not (3D or 2D input). Default is False.
        Returns
        -------
        None
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.name = name
        self.batched_input = batched_input
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        self.optimizer_type = optimizer_type

        # Initialize layers only if input_size is provided
        if self.input_size is not None:
            self._initialize()

    def reset_hyperparameter(self, n_hidden_size, n_hidden_layers, learning_rate, activation, optimizer_type):
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate
        self.activation = self.activation_map[activation]()
        self.optimizer_type = optimizer_type
        self.scaler = None
        if self.input_size is not None:
            self._initialize()

    def _initialize(self):
        """Initialize the layers of the neural network."""
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size
        self.rnn = nn.RNN(self.input_size, self.n_hidden_size, self.n_hidden_layers, batch_first=self.batched_input)
        self.fc = nn.Linear(self.n_hidden_size, self.output_size)
        self.to(self.device)

    def forward(self, x):
        if self.input_size is None:
            self.input_size = x.shape[-1]
            if self.n_hidden_size is None:
                self.n_hidden_size = self.input_size
            self._initialize()
            self.to(self.device)

        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        x = x.to(self.device)
        if not x.is_contiguous():
            x = x.contiguous()

        batch_size = x.size(1)
        h0 = torch.zeros(self.n_hidden_layers, batch_size, self.n_hidden_size, device=self.device)

        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

    @staticmethod
    def get_reference_model(input_size=None):
        """
        Get a reference neural network with specified input size.

        This function initializes and returns a neural network model with the given input size.
        The output size and hidden size are set to 1 and the input size, respectively.

        Parameters
        ----------
        input_size : int
            The number of input features for the neural network.
            Default is None. -> Input will be set at runtime.
        Returns
        -------
        Net
            An instance of the Net class configured with the specified input size.
        """
        return RNN(input_size, 1, input_size)

class LSTM(mb.BaseTorchModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=1, activation=nn.ReLU,
                 learning_rate=0.001, name="LSTM", batched_input=False):
        """
        Initializes a configurable recurrent neural network.

        Parameters
        ----------
        input_size : int, optional
            The number of input features. If None, it will be set during the first training call.
        output_size : int
            The number of output features.
        n_hidden_size : int, optional
            The number of features in each hidden layer. If None, it will be set to the input size during the first training call.
        n_hidden_layers : int
            The number of hidden layers in the network.
        activation : torch.nn.Module, optional
            The activation function to be used in the hidden layers. The default is nn.ReLU.
        name: str
            The name of the model.
        batched_input : bool
            Is the input batched or not (3D or 2D input). Default is False.
        Returns
        -------
        None
        """
        super(LSTM, self).__init__()
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
        self.batched_input = False

        # Initialize layers only if input_size is provided
        if self.input_size is not None:
            self._initialize()

    def reset_hyperparameter(self, n_hidden_size, n_hidden_layers, learning_rate, activation, optimizer_type):
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate
        self.activation = self.activation_map[activation]()
        self.optimizer_type = optimizer_type

    def _initialize(self):
        """Initialize the layers of the neural network."""
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size
        self.rnn = nn.LSTM(self.input_size, self.n_hidden_size, self.n_hidden_layers, batch_first=self.batched_input)
        self.fc = nn.Linear(self.n_hidden_size, self.output_size)
        self.to(self.device)

    def forward(self, x):
        """
        Defines the forward pass of the recurrent neural network (e.g. LSTM).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch_size, seq_len, input_size] or [seq_len, input_size].

        Returns
        -------
        torch.Tensor
            Output tensor from the neural network.
        """
        # Initialize model parameters if not yet initialized
        if self.input_size is None:
            self.input_size = x.shape[-1]
            if self.n_hidden_size is None:
                self.n_hidden_size = self.input_size
            self._initialize()
            self.to(self.device)

        # If input is 2D (single sample), add batch dimension
        if x.dim() == 2:  # [seq_len, input_size]
            x = x.unsqueeze(0)  # → [1, seq_len, input_size]

        x = x.to(self.device)
        x = x.contiguous()  # Important for cuDNN

        batch_size = x.size(1)

        h0 = torch.zeros(self.n_hidden_layers, batch_size, self.n_hidden_size, device=self.device)
        c0 = torch.zeros(self.n_hidden_layers, batch_size, self.n_hidden_size, device=self.device)

        # Forward pass through LSTM
        out, _ = self.rnn(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out)

        return out
    @staticmethod
    def get_reference_model(input_size=None):
        """
        Get a reference neural network with specified input size.

        This function initializes and returns a neural network model with the given input size.
        The output size and hidden size are set to 1 and the input size, respectively.

        Parameters
        ----------
        input_size : int
            The number of input features for the neural network.
            Default is None. -> Input will be set at runtime.
        Returns
        -------
        LSTM
            An instance of the Net class configured with the specified input size.
        """
        return LSTM(input_size, 1, input_size)

class GRU(mb.BaseTorchModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=1, activation=nn.ReLU,
                 learning_rate=0.001, name="GRU", batched_input=False):
        """
        Initializes a configurable recurrent neural network.

        Parameters
        ----------
        input_size : int, optional
            The number of input features. If None, it will be set during the first training call.
        output_size : int
            The number of output features.
        n_hidden_size : int, optional
            The number of features in each hidden layer. If None, it will be set to the input size during the first training call.
        n_hidden_layers : int
            The number of hidden layers in the network.
        activation : torch.nn.Module, optional
            The activation function to be used in the hidden layers. The default is nn.ReLU.
        name: str
            The name of the model.
        batched_input : bool
            Is the input batched or not (3D or 2D input). Default is False.
        Returns
        -------
        None
        """
        super(GRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation()
        self.learning_rate = learning_rate
        self.name = name
        self.batched_input = batched_input
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None

        # Initialize layers only if input_size is provided
        if self.input_size is not None:
            self._initialize()

    def reset_hyperparameter(self, n_hidden_size, n_hidden_layers, learning_rate, activation, optimizer_type):
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate
        self.activation = self.activation_map[activation]()
        self.optimizer_type = optimizer_type

    def _initialize(self):
        """Initialize the layers of the neural network."""
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size
        self.rnn = nn.GRU(self.input_size, self.n_hidden_size, self.n_hidden_layers, batch_first=self.batched_input)
        self.fc = nn.Linear(self.n_hidden_size, self.output_size)
        self.to(self.device)

    def forward(self, x):
        """
        Defines the forward pass of the recurrent neural network (vanilla RNN).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch_size, seq_len, input_size] or [seq_len, input_size].

        Returns
        -------
        torch.Tensor
            Output tensor from the neural network.
        """
        # Initialize model parameters if not yet initialized
        if self.input_size is None:
            self.input_size = x.shape[-1]
            if self.n_hidden_size is None:
                self.n_hidden_size = self.input_size
            self._initialize()
            self.to(self.device)

        # Handle non-batched input
        if x.dim() == 2:  # [seq_len, input_size]
            x = x.unsqueeze(0)  # → [1, seq_len, input_size]

        x = x.to(self.device).contiguous()

        batch_size = x.size(1)

        # Initialize hidden state: [n_layers, batch_size, hidden_size]
        h0 = torch.zeros(self.n_hidden_layers, batch_size, self.n_hidden_size, device=self.device)

        # Forward pass through RNN
        out, _ = self.rnn(x, h0)

        # Fully connected layer on all time steps
        out = self.fc(out)

        return out

    @staticmethod
    def get_reference_model(input_size=None):
        """
        Get a reference neural network with specified input size.

        This function initializes and returns a neural network model with the given input size.
        The output size and hidden size are set to 1 and the input size, respectively.

        Parameters
        ----------
        input_size : int
            The number of input features for the neural network.
            Default is None. -> Input will be set at runtime.
        Returns
        -------
        GRU
            An instance of the Net class configured with the specified input size.
        """
        return GRU(input_size, 1, input_size)

class TorchAttention(mb.BaseTorchModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, activation=nn.ReLU, learning_rate=0.0001, name="Neural_Net_with_Attention"):
        """
        Initializes a neural network with a normal layer, an attention layer, and another normal layer.

        Parameters
        ----------
        input_size : int, optional
            The number of input features. If None, it will be set during the first training call.
        output_size : int
            The number of output features.
        n_hidden_size : int, optional
            The number of features in each hidden layer. If None, it will be set to the input size during the first training call.
        learning_rate : float
            The learning rate for the optimizer.
        name : str
            The name of the model.
        """
        super(TorchAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_size = n_hidden_size
        self.activation = activation()
        self.n_hidden_layers = 1
        self.learning_rate = learning_rate
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None

        # Initialize layers only if input_size is provided
        if self.input_size is not None:
            self._initialize()

    def _initialize(self):
        """Initialize the layers of the neural network."""
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size

        # First normal layer
        self.fc1 = nn.Linear(self.input_size, self.n_hidden_size)

        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=self.n_hidden_size, num_heads=1)

        # Second normal layer
        self.fc2 = nn.Linear(self.n_hidden_size, self.n_hidden_size)

        # Output layer
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

        # First normal layer with ReLU activation
        x = self.activation(self.fc1(x))

        # Reshape for attention layer
        x = x.unsqueeze(0)  # Add sequence length dimension
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)  # Remove sequence length dimension

        # Second normal layer with ReLU activation
        x = self.activation(self.fc2(x))

        # Output layer
        x = self.fc3(x)
        return x

class TorchTransformer(mb.BaseTorchModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, activation=nn.ReLU, learning_rate=0.0001, name="Neural_Net_Transformer"):
        """
        Initializes a neural network with a normal layer, a transformer layer, and another normal layer.

        Parameters
        ----------
        input_size : int, optional
            The number of input features. If None, it will be set during the first training call.
        output_size : int
            The number of output features.
        n_hidden_size : int, optional
            The number of features in each hidden layer. If None, it will be set to the input size during the first training call.
        learning_rate : float
            The learning rate for the optimizer.
        name : str
            The name of the model.
        """
        super(TorchTransformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = 1
        self.activation = activation()
        self.learning_rate = learning_rate
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None

        # Initialize layers only if input_size is provided
        if self.input_size is not None:
            self._initialize()

    def _initialize(self):
        """Initialize the layers of the neural network."""
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size

        # First normal layer
        self.fc1 = nn.Linear(self.input_size, self.n_hidden_size)

        # Transformer layer
        self.transformer = nn.Transformer(
            d_model=self.n_hidden_size,
            nhead=1,
            num_encoder_layers=1,
            num_decoder_layers=1
        )

        # Second normal layer
        self.fc2 = nn.Linear(self.n_hidden_size, self.n_hidden_size)

        # Output layer
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

        # First normal layer with ReLU activation
        x = self.activation(self.fc1(x))

        # Reshape for transformer layer
        x = x.unsqueeze(0)  # Add sequence length dimension
        x = self.transformer(x, x)
        x = x.squeeze(0)  # Remove sequence length dimension

        # Second normal layer with ReLU activation
        x = self.activation(self.fc2(x))

        # Output layer
        x = self.fc3(x)
        return x

class QuantileIdNetModel(Net):
    def __init__(self, input_size, output_size, n_neurons, n_layers, activation=nn.ReLU, output_distribution='uniform', learning_rate=0.0001, name = 'Net_Q_Id'):
        """
        Initializes a configurable neural network with Quantile + Id scaling.

        Parameters
        ----------
        input_size : int
            The number of original input features (before scaling).
        output_size : int
            The number of output features.
        n_neurons : int
            The number of neurons in each hidden layer.
        n_layers : int
            The number of hidden layers in the network.
        activation : torch.nn.Module, optional
            The activation function to be used in the hidden layers. The default is nn.ReLU.
        output_distribution : str, optional
            The output distribution for the QuantileTransformer. Default is 'uniform'.
        """
        # Adjust input size to account for doubled features due to Quantile + Id scaling
        super(QuantileIdNetModel, self).__init__(input_size * 2, output_size, n_neurons, n_layers, activation, learning_rate, name)
        self.output_distribution = output_distribution

    def scale_data(self, X):
        """
        Scale the input data using QuantileTransformer and keep a copy of each original feature.

        Parameters:
        X (Tensor): The input data.

        Returns:
        Tensor: The scaled input data.
        """
        if self.scaler is None:
            self.scaler = QuantileTransformer(output_distribution=self.output_distribution)
            self.scaler.fit(X)
        X_quantile = self.scaler.transform(X)
        X_scaled = np.hstack((X_quantile, X))
        return X_scaled

class RiemannQuantileClassifierNet(Net):
    def __init__(self, input_size, output_size=1, n_neurons=64, n_layers=3, activation=nn.ReLU, min_bins=8,
                 max_bins=64, learning_rate=0.0001, name = 'RiemannQuantileClassifierNet'):
        super().__init__(input_size, max_bins, n_neurons, n_layers, activation)
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.bins = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.name = name
    def discretize_targets(self, y):
        if self.bins is None:
            n_unique = len(np.unique(y))
            adaptive_bins = min(self.max_bins, max(self.min_bins, n_unique // 2))
            self.bins = np.quantile(y, q=np.linspace(0, 1, adaptive_bins + 1))
            self.bins = np.unique(self.bins)
            self.n_bins = len(self.bins) - 1
            self.output_size = self.n_bins
        y_digitized = np.digitize(y, bins=self.bins[1:-1])
        return y_digitized

    def criterion(self, y_target, y_pred_logits):
        y_target = y_target.long().squeeze()
        return self.loss_fn(y_pred_logits, y_target)

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=100, patience=20,
                    draw_loss=False, epsilon=0.0001, trial=None, n_outlier=12):
        print(self.device)

        y_train_discrete = self.discretize_targets(y_train)
        y_val_discrete = self.discretize_targets(y_val)

        X_train_scaled = self.scale_data(X_train)
        X_val_scaled = self.scale_data(X_val)

        X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train_discrete, dtype=torch.long).to(self.device)
        y_val_tensor = torch.tensor(y_val_discrete, dtype=torch.long).to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_error = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(X_train)
            # TODO: Prüfen was stimmt
            # logits = self.predict(X_tensor)
            # loss = self.loss_fn(logits, torch.tensor(y_target_discrete))
            loss = self.criterion(y_train_tensor, y_pred)
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                y_val_pred = self(X_val)
                val_error = self.criterion(y_val_tensor, y_val_pred).item()

                if val_error < best_val_error - epsilon:
                    best_val_error = val_error
                    best_model_state = self.state_dict()
                    patience_counter = 0
                elif epoch > (n_epochs / 10) and epoch > 10:
                    patience_counter += 1

                scheduler.step(val_error)

                if trial is not None:
                    trial.report(val_error, step=epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            print(
                f'{self.name}: Epoch {epoch + 1}/{n_epochs}, Train Loss; {loss:.4f} Val Error: {val_error:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.load_state_dict(best_model_state)
        return best_val_error

    def test_model(self, X, y_target, criterion_test=None):
        if criterion_test is None:
            criterion_test = nn.MSELoss()
        X_scaled = self.scale_data(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.eval()
        with torch.no_grad():
            logits = self.predict(X_tensor)
            pred_probs = torch.softmax(logits, dim=1).cpu().numpy()

        if self.bins is not None:
            bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
            y_pred = np.sum(pred_probs * bin_centers[None, :], axis=1)
        else:
            y_pred = np.argmax(pred_probs, axis=1)

        y_target_discrete = self.discretize_targets(y_target)
        loss = criterion_test(torch.tensor(pred_probs), torch.tensor(y_target_discrete))

        return loss.item(), y_pred

# ToDo: Im gegensatz zu RiemanQuantileClassifier funktioniert das noch nicht, Theoretisch müsste normierung ergebniss aber verbessern
class QuantileIdRiemannClassifierNet(Net):
    def __init__(self, input_size, output_size=1, n_neurons=64, n_layers=3, activation=nn.ReLU, min_bins=8, max_bins=64,
                 output_distribution='uniform'):
        super().__init__(input_size * 2, max_bins, n_neurons, n_layers, activation)
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.output_distribution = output_distribution
        self.bins = None
        self.scaler = None
        self.loss_fn = nn.CrossEntropyLoss()

    def scale_data(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.scaler is None:
            from sklearn.preprocessing import QuantileTransformer
            self.scaler = QuantileTransformer(output_distribution=self.output_distribution)
            self.scaler.fit(X)
        X_quantile = self.scaler.transform(X)
        return np.hstack((X_quantile, X))

    def discretize_targets(self, y):
        if self.bins is None:
            n_unique = len(np.unique(y))
            adaptive_bins = min(self.max_bins, max(self.min_bins, n_unique // 2))
            self.bins = np.quantile(y, q=np.linspace(0, 1, adaptive_bins + 1))
            self.bins = np.unique(self.bins)
            self.n_bins = len(self.bins) - 1
            self.output_size = self.n_bins
        y_digitized = np.digitize(y, bins=self.bins[1:-1])
        return y_digitized

    def criterion(self, y_target, y_pred_logits):
        return self.loss_fn(y_pred_logits, y_target.long().squeeze())

    def train_model(self, X_train, y_train, X_val, y_val, learning_rate=0.0001, n_epochs=100, patience=20,
                    draw_loss=False, epsilon=0.0001, trial=None, n_outlier=12):
        print(self.device)

        y_train_discrete = self.discretize_targets(y_train)
        y_val_discrete = self.discretize_targets(y_val)

        X_train_scaled = self.scale_data(X_train)
        X_val_scaled = self.scale_data(X_val)

        X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train_discrete, dtype=torch.long).to(self.device)
        y_val_tensor = torch.tensor(y_val_discrete, dtype=torch.long).to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        best_val_error = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(X_train)
            loss = self.criterion(y_train_tensor, y_pred)
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                y_val_pred = self(X_val)
                val_error = self.criterion(y_val_tensor, y_val_pred).item()

                if val_error < best_val_error - epsilon:
                    best_val_error = val_error
                    best_model_state = self.state_dict()
                    patience_counter = 0
                elif epoch > (n_epochs / 10) and epoch > 10:
                    patience_counter += 1

                scheduler.step(val_error)

                if trial is not None:
                    trial.report(val_error, step=epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            print(
                f'Epoch {epoch + 1}/{n_epochs}, Val Error: {val_error:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.load_state_dict(best_model_state)
        return best_val_error

    def test_model(self, X, y_target, criterion_test=None):
        if criterion_test is None:
            criterion_test = self.criterion
        X_scaled = self.scale_data(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.eval()
        with torch.no_grad():
            logits = self.predict(X_tensor)
            pred_probs = torch.softmax(logits, dim=1).cpu().numpy()

        if self.bins is not None:
            bin_centers = (self.bins[:-1] + self.bins[1:]) / 2
            y_pred = np.sum(pred_probs * bin_centers[None, :], axis=1)
        else:
            y_pred = np.argmax(pred_probs, axis=1)

        y_target_discrete = self.discretize_targets(y_target)
        loss = criterion_test(torch.tensor(pred_probs), torch.tensor(y_target_discrete))

        return loss.item(), y_pred

class PiNNErd(Net):
    def __init__(self, *args, name="PiNN_Erd", penalty_weight=1, optimizer_type='adam', theta_init=None, **kwargs):
        super(PiNNErd, self).__init__(*args, **kwargs)
        self.name = name
        self.penalty_weight = penalty_weight
        self.optimizer_type = optimizer_type
        self.theta_init = theta_init
        # Neue Parameter-Matrix für alle Achsen
        if theta_init is not None:
            if theta_init.shape != (4, 5):
                raise ValueError("theta_init must have shape (4, 5)")
            self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        else:
            self.theta = nn.Parameter(torch.zeros(4, 5, dtype=torch.float32))  # Achsen: x, y, z, sp

    def _initialize(self):
        """Initialize the layers of the neural network."""
        self.scaler = None
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size
        self.fc1 = nn.Linear(self.input_size, self.n_hidden_size)
        self.fcs = nn.ModuleList([nn.Linear(self.n_hidden_size, self.n_hidden_size) for _ in range(self.n_hidden_layers)])
        self.fc3 = nn.Linear(self.n_hidden_size, self.output_size)
        self.to(self.device)

        if self.theta_init is not None:
            self.theta = nn.Parameter(torch.tensor(self.theta_init, dtype=torch.float32))
        else:
            self.theta = nn.Parameter(torch.zeros(4, 5, dtype=torch.float32))  # Achsen: x, y, z, sp

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Überschreibt das Training, um x_input an die Loss-Funktion zu übergeben.
        """
        original_criterion = self.criterion

        # Patch `self.criterion` temporär für das Training
        current_input = self.scaled_to_tensor(X_train)

        def custom_train_model(*args, **kwargs):
            nonlocal current_input
            original_train_model = super(PiNNErd, self).train_model

            def patched_criterion(y_target, y_pred):
                return original_criterion(y_target, y_pred, x_input=current_input)

            self.criterion = patched_criterion
            result = original_train_model(*args, **kwargs)
            self.criterion = original_criterion  # Restore
            return result

        # Trick: train_model aus BaseNetModel verwendet `self.criterion`
        return custom_train_model(X_train, y_train, X_val, y_val, **kwargs)

    def criterion(self, y_target, y_pred, x_input=None):
        criterion = nn.MSELoss()
        mse_loss = criterion(y_target.squeeze(), y_pred.squeeze())

        if x_input is not None and y_pred.requires_grad or y_target.requires_grad:
            x_input = x_input.clone().detach().requires_grad_(True)
            y_pred_physics = self(x_input)

            dy_dx = torch.autograd.grad(
                outputs=y_pred_physics,
                inputs=x_input,
                grad_outputs=torch.ones_like(y_pred_physics),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            if x_input.size(1) == 13:
                # Features extrahieren
                a = x_input[:, 0:4]  # [a_sp, a_x, a_y, a_z]
                f = x_input[:, 4:8]  # [f_sp, f_x, f_y, f_z]
                mrr = x_input[:, 8].unsqueeze(1)  # (N, 1)
                v = x_input[:, 9:13]  # [v_sp, v_x, v_y, v_z]

                ones = torch.ones_like(mrr)  # bias-term
                input_features = torch.cat([a, v, f, mrr, ones], dim=1)  # (N, 13)

                # Ableitungen extrahieren: d/d[a_sp, a_x, a_y, a_z, ..., v_z]
                dy_da = dy_dx[:, 0:4]
                dy_dv = dy_dx[:, 9:13]
                dy_df = dy_dx[:, 4:8]
                dy_dmrr = dy_dx[:, 8].unsqueeze(1)  # (N, 1)

                # Constraint-Berechnung für jede Achse separat
                constraint = []
                for i in range(4):  # 4 Achsen: sp, x, y, z
                    deriv = dy_da[:, i] + dy_dv[:, i] + dy_df[:, i] + dy_dmrr.squeeze(1)
                    influences = (
                            a[:, i] * self.theta[i, 0] +
                            v[:, i] * self.theta[i, 1] +
                            f[:, i] * self.theta[i, 2] +
                            mrr.squeeze(1) * self.theta[i, 3] +
                            self.theta[i, 4]  # bias
                    )
                    constraint_i = deriv - influences
                    constraint.append(constraint_i.unsqueeze(1))

                constraint = torch.cat(constraint, dim=1)  # (N, 4)
                penalty = torch.mean(constraint ** 2)
                return mse_loss + self.penalty_weight * penalty

            elif x_input.size(1) == 4:
                # Alte Minimalform (x-Komponenten only)
                a = x_input[:, 0]
                f = x_input[:, 1]
                mrr = x_input[:, 2]
                v = x_input[:, 3]

                dy_da = dy_dx[:, 0]
                dy_df = dy_dx[:, 1]
                dy_dmrr = dy_dx[:, 2]
                dy_dv = dy_dx[:, 3]

                deriv = dy_da + dy_dv + dy_df + dy_dmrr
                influences = (
                        a * self.theta[1, 0] +
                        v * self.theta[1, 1] +
                        f * self.theta[1, 2] +
                        mrr * self.theta[1, 3] +
                        self.theta[1, 4]
                )
                penalty = torch.mean((deriv - influences) ** 2)
                return mse_loss + self.penalty_weight * penalty

            else:
                throw_error('x_input hat die falsche Größe')

        return mse_loss

    def get_documentation(self):
        documentation = {"hyperparameters": {
            "learning_rate": self.learning_rate,
            "n_hidden_size": self.n_hidden_size,
            "n_hidden_layers": self.n_hidden_layers,
            "n_activation_function": self.activation.__class__.__name__,
            "optimizer_type": self.optimizer_type,
            "penalty_weight": self.penalty_weight,
            #"theta_init": self.theta_init.tolist(),
        }}
        return documentation

class PiNNNaiveLinear(Net):
    def __init__(self, *args, name="PiNN_Naive_Linear", penalty_weight=1, optimizer_type='adam', learning_rate=0.001, theta_init=None,**kwargs):
        super(PiNNNaiveLinear, self).__init__(*args, learning_rate=learning_rate, **kwargs)
        self.name = name
        self.penalty_weight = penalty_weight
        self.optimizer_type = optimizer_type
        self.theta_init = theta_init
        # Neue Parameter-Matrix für alle Achsen
        if theta_init is not None:
            if theta_init.shape != (4, 2):
                raise ValueError("theta_init must have shape (4, 3)")
            self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        else:
            self.theta = nn.Parameter(torch.zeros(4, 2, dtype=torch.float32))  # Achsen: x, y, z, sp

    def criterion(self, y_target, y_pred, x_input=None):
        #ToDo: Modularer gestalten
        criterion = nn.MSELoss()
        mse_loss = criterion(y_target.squeeze(), y_pred.squeeze())

        if x_input is not None and y_pred.requires_grad or y_target.requires_grad:
            x_input = x_input.clone().detach().requires_grad_(True)
            y_pred_physics = self(x_input)

            dy_dx = torch.autograd.grad(
                outputs=y_pred_physics,
                inputs=x_input,
                grad_outputs=torch.ones_like(y_pred_physics),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            if x_input.size(1) == 13:
                # Features extrahieren
                a = x_input[:, 0:4]  # [a_sp, a_x, a_y, a_z]
                f = x_input[:, 4:8]  # [f_sp, f_x, f_y, f_z]
                mrr = x_input[:, 8].unsqueeze(1)  # (N, 1)
                v = x_input[:, 9:13]  # [v_sp, v_x, v_y, v_z]

                ones = torch.ones_like(mrr)  # bias-term
                input_features = torch.cat([a, v, f, mrr, ones], dim=1)  # (N, 13)

                # Ableitungen extrahieren: d/d[a_sp, a_x, a_y, a_z, ..., v_z]
                dy_da = dy_dx[:, 0:4]
                dy_dv = dy_dx[:, 9:13]
                dy_df = dy_dx[:, 4:8]
                dy_dmrr = dy_dx[:, 8].unsqueeze(1)  # (N, 1)

                # Constraint-Berechnung für jede Achse separat
                constraint = []
                deriv =  dy_dv + dy_df
                influences = (
                        self.theta[:, 0] +
                        v * self.theta[:, 1]
                )
                constraint_i = deriv - influences
                constraint.append(constraint_i.unsqueeze(1))

                constraint = torch.cat(constraint, dim=1)  # (N, 4)
                penalty = torch.mean(constraint ** 2)
                return mse_loss + self.penalty_weight * penalty

            elif x_input.size(1) == 4:
                # Alte Minimalform (x-Komponenten only)
                a = x_input[:, 0]
                f = x_input[:, 1]
                mrr = x_input[:, 2]
                v = x_input[:, 3]

                dy_da = dy_dx[:, 0]
                dy_df = dy_dx[:, 1]
                dy_dmrr = dy_dx[:, 2]
                dy_dv = dy_dx[:, 3]

                deriv = dy_dv + dy_df
                influences = (
                        self.theta[1, 0] +
                        v * self.theta[1, 1]# ToDo; auf abs ändern
                )
                penalty = torch.mean((deriv - influences) ** 2)
                return mse_loss + self.penalty_weight * penalty

            else:
                throw_error(self, 'x_input hat die falsche Größe')

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

        current_input = self.scaled_to_tensor(X_train)

        def custom_train_model(*args, **kwargs):
            nonlocal current_input
            original_train_model = super(PiNNNaiveLinear, self).train_model

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

class PiNNFriction(Net):
    def __init__(self, *args, name="PiNNFriction", penalty_weight=1, optimizer_type='adam', learning_rate=0.001, theta_init=None,**kwargs):
        super(PiNNFriction, self).__init__(*args, learning_rate=learning_rate, **kwargs)
        self.name = name
        self.penalty_weight = penalty_weight
        self.optimizer_type = optimizer_type
        self.theta_init = theta_init
        self.input_head = hdata.HEADER_x
        # Neue Parameter-Matrix für alle Achsen
        if theta_init is not None:
            if theta_init.shape != (4, 5):
                raise ValueError("theta_init must have shape (4, 5)")
            self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        else:
            self.theta = nn.Parameter(torch.zeros(4, 5, dtype=torch.float32))  # Achsen: x, y, z, sp

    def get_indices_by_prefix(self, header, prefix):
        return [index for index, item in enumerate(header) if item.startswith(prefix)]

    def criterion(self, y_target, y_pred, x_input=None):

        criterion = nn.MSELoss()
        mse_loss = criterion(y_target.squeeze(), y_pred.squeeze())

        if x_input is not None and y_pred.requires_grad or y_target.requires_grad:
            x_input = x_input.clone().detach().requires_grad_(True)
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

            # Extrahiere die Indizes und die entsprechenden Daten
            indices_a = self.get_indices_by_prefix(self.input_head, f'a_{axis}')
            indices_f = self.get_indices_by_prefix(self.input_head, f'f_{axis}')
            indices_v = self.get_indices_by_prefix(self.input_head, f'v_{axis}')
            indices_z = self.get_indices_by_prefix(self.input_head, f'z_{axis}')#

            # Kombiniere alle Indizes, die wir nicht in dy_remaining wollen
            all_excluded_indices = set(indices_a + indices_f + indices_v + indices_z)

            # Erstelle eine Maske für die verbleibenden Indizes
            remaining_indices = [i for i in range(dy_dx.shape[1]) if i not in all_excluded_indices]

            # Extrahiere die verbleibenden Teile von dy_dx
            dy_remaining = dy_dx[:, remaining_indices]

            # Die bereits extrahierten Teile
            dy_da = dy_dx[:, indices_a]
            dy_df = dy_dx[:, indices_f]
            dy_dv = dy_dx[:, indices_v]
            dy_dz = dy_dx[:, indices_z]

            a = x_input[:, indices_a]
            f = x_input[:, indices_f]
            v = x_input[:, indices_v]
            z = x_input[:, indices_z]

            # Constraint-Berechnung für jede Achse separat
            constraint = []
            deriv =  dy_dv + dy_df + dy_da + dy_dz
            dim = v.shape[1]
            influences = (
                    self.theta[1, 0] +
                    v * self.theta[:dim, 1] +
                    f * self.theta[:dim, 2] +
                    a * self.theta[:dim, 3] +
                    z * self.theta[:dim, 4]
            )
            constraint_i = deriv - influences
            constraint.append(constraint_i.unsqueeze(1))

            constraint = torch.cat(constraint, dim=1)  # (N, 4)
            penalty = torch.mean(constraint ** 2)  + 1/10 * torch.mean(dy_remaining ** 2)
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
            original_train_model = super(PiNNFriction, self).train_model

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

class PiNNNaive(Net):
    def __init__(self, *args, name="PiNN_Naive", penalty_weight=1, optimizer_type='adam', learning_rate=0.001,  theta_init=None, **kwargs):
        super(PiNNNaive, self).__init__(*args,learning_rate=learning_rate, **kwargs)
        self.name = name
        self.penalty_weight = penalty_weight
        self.optimizer_type = optimizer_type
        self.theta_init = theta_init
        # Neue Parameter-Matrix für alle Achsen
        if theta_init is not None:
            if theta_init.shape != (4, 3):
                raise ValueError("theta_init must have shape (4, 3)")
            self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        else:
            self.theta = nn.Parameter(torch.zeros(4, 3, dtype=torch.float32))  # Achsen: x, y, z, sp

    def sigmoid_stable(self, x):
        """Verwendet die eingebaute, numerisch stabile Sigmoid-Funktion von PyTorch."""
        # Sicherstellen, dass x ein Tensor ist
        if isinstance(x, list):
            x = torch.stack(x) if len(x) > 1 else x[0]

        # Verschieben und Clipping
        x_shifted = x + self.theta[:, 2]
        x_clipped = torch.clamp(x_shifted, min=-50, max=50)

        # PyTorch's eingebaute Sigmoid-Funktion verwenden
        sigmoid_part = torch.sigmoid(x_clipped)

        # Finales Clipping
        result = torch.clamp(sigmoid_part, min=-1e6, max=1e6)

        return result

    def criterion(self, y_target, y_pred, x_input=None):
        #ToDo: Modularer gestalten
        criterion = nn.MSELoss()
        mse_loss = criterion(y_target.squeeze(), y_pred.squeeze())

        if x_input is not None and y_pred.requires_grad or y_target.requires_grad:
            x_input = x_input.clone().detach().requires_grad_(True)
            y_pred_physics = self(x_input)

            dy_dx = torch.autograd.grad(
                outputs=y_pred_physics,
                inputs=x_input,
                grad_outputs=torch.ones_like(y_pred_physics),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            if x_input.size(1) == 13:
                # Features extrahieren
                a = x_input[:, 0:4]  # [a_sp, a_x, a_y, a_z]
                f = x_input[:, 4:8]  # [f_sp, f_x, f_y, f_z]
                mrr = x_input[:, 8].unsqueeze(1)  # (N, 1)
                v = x_input[:, 9:13]  # [v_sp, v_x, v_y, v_z]

                ones = torch.ones_like(mrr)  # bias-term
                input_features = torch.cat([a, v, f, mrr, ones], dim=1)  # (N, 13)

                # Ableitungen extrahieren: d/d[a_sp, a_x, a_y, a_z, ..., v_z]
                dy_da = dy_dx[:, 0:4]
                dy_dv = dy_dx[:, 9:13]
                dy_df = dy_dx[:, 4:8]
                dy_dmrr = dy_dx[:, 8].unsqueeze(1)  # (N, 1)

                # Constraint-Berechnung für jede Achse separat
                constraint = []
                deriv = dy_da + dy_dv+ dy_df
                influences = (
                        self.theta[:, 0] +
                        self.sigmoid_stable(v) * self.theta[:, 1]
                )
                constraint_i = deriv - influences
                constraint.append(constraint_i.unsqueeze(1))



                constraint = torch.cat(constraint, dim=1)  # (N, 4)
                penalty = torch.mean(constraint ** 2)
                return mse_loss + self.penalty_weight * penalty

            elif x_input.size(1) == 4:
                # Alte Minimalform (x-Komponenten only)
                a = x_input[:, 0]
                f = x_input[:, 1]
                mrr = x_input[:, 2]
                v = x_input[:, 3]

                dy_da = dy_dx[:, 0]
                dy_df = dy_dx[:, 1]
                dy_dmrr = dy_dx[:, 2]
                dy_dv = dy_dx[:, 3]

                deriv = dy_dv + dy_df
                influences = (
                        self.theta[1, 0] +
                        v * self.theta[1, 1] + # ToDo; auf abs ändern
                        self.theta[1, 2]
                )
                penalty = torch.mean((deriv - influences) ** 2)
                return mse_loss + self.penalty_weight * penalty

            else:
                throw_error(self, 'x_input hat die falsche Größe')

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

        current_input = self.scaled_to_tensor(X_train)

        def custom_train_model(*args, **kwargs):
            nonlocal current_input
            original_train_model = super(PiNNNaive, self).train_model

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
