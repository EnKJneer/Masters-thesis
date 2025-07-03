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
class Net(mb.BaseNetModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=1, activation=nn.ReLU,
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
        self.activation = activation()
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
        return x

class RNN(mb.BaseNetModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=1, activation=nn.ReLU,
                 learning_rate=0.001, name="Recurrent_Neural_Net", batched_input=False):
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

class LSTM(mb.BaseNetModel):
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
        self.batched_input = batched_input
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

class GRU(mb.BaseNetModel):
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

class PartialRnn(mb.BaseNetModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=1, n_rnn_layer=1, activation=nn.PReLU, learning_rate=0.01, name="Partial_Recurrent_Neural_Net"):
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

        Returns
        -------
        None
        """
        super(PartialRnn, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.n_rnn_layer = n_rnn_layer
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
        self.fc_in = nn.Linear(self.input_size, self.n_hidden_size)
        self.fcs = nn.ModuleList([nn.Linear(self.n_hidden_size, self.n_hidden_size) for _ in range(self.n_hidden_layers)])
        self.n_rnn_size = 2 * self.output_size
        self.fc_connection = nn.Linear(self.n_hidden_size, self.n_rnn_size)
        self.rnn = nn.RNN(self.n_rnn_size, self.n_rnn_size, self.n_rnn_layer)
        self.fc_end_connection = nn.Linear(self.n_rnn_size, self.n_rnn_size)
        self.fc_out = nn.Linear(self.n_rnn_size, self.output_size)
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
            self.to(self.device)

        x = self.activation(self.fc_in(x))  # apply the activation function
        for fc in self.fcs:
            x = self.activation(fc(x))  # apply the activation function
        x = self.fc_connection(x)
        if x.dim() == 2:
            # Non batched input
            h0 = torch.zeros(self.n_rnn_layer, self.n_rnn_size).to(self.device)
        elif x.dim() == 3:
            # batched input
            h0 = torch.zeros(self.n_rnn_layer, x.size(0), 1).to(self.device)

        # Forward pass through RNN layer
        x, _ = self.rnn(x, h0)
        x = self.activation(self.fc_end_connection(x))
        x = self.fc_out(x)

        return x

class PartialGRU(mb.BaseNetModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=1, n_rnn_layer=1, activation=nn.PReLU, learning_rate=0.01, name="Partial_GRU_Neural_Net"):
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

        Returns
        -------
        None
        """
        super(PartialGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.n_rnn_layer = n_rnn_layer
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
        self.fc_in = nn.Linear(self.input_size, self.n_hidden_size)
        self.fcs = nn.ModuleList([nn.Linear(self.n_hidden_size, self.n_hidden_size) for _ in range(self.n_hidden_layers)])
        self.n_rnn_size = 2 * self.output_size
        self.fc_connection = nn.Linear(self.n_hidden_size, self.n_rnn_size)
        self.rnn = nn.GRU(self.n_rnn_size, self.n_rnn_size, self.n_rnn_layer)
        self.fc_end_connection = nn.Linear(self.n_rnn_size, self.n_rnn_size)
        self.fc_out = nn.Linear(self.n_rnn_size, self.output_size)
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
            self.to(self.device)

        x = self.activation(self.fc_in(x))  # apply the activation function
        for fc in self.fcs:
            x = self.activation(fc(x))  # apply the activation function
        x = self.fc_connection(x)
        if x.dim() == 2:
            # Non batched input
            h0 = torch.zeros(self.n_rnn_layer, self.n_rnn_size).to(self.device)
        elif x.dim() == 3:
            # batched input
            h0 = torch.zeros(self.n_rnn_layer, x.size(0), 1).to(self.device)

        # Forward pass through RNN layer
        x, _ = self.rnn(x, h0)
        x = self.activation(self.fc_end_connection(x))
        x = self.fc_out(x)

        return x

class NetAttention(mb.BaseNetModel):
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
        super(NetAttention, self).__init__()
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

class NetTransformer(mb.BaseNetModel):
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
        super(NetTransformer, self).__init__()
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

class PiNN(Net):
    def __init__(self, *args, name="PiNN_Erd", penalty_weight=1, optimizer_type='adam', c_1=-6.23e-6, c_2=2.27e-7, c_3=-2.7e-6, **kwargs):
        super(PiNN, self).__init__(*args, **kwargs)
        self.name = name
        self.penalty_weight = penalty_weight
        self.optimizer_type = optimizer_type
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_3 =c_3

    def criterion(self, y_target, y_pred, x_input=None):
        """
        Loss-Funktion mit zusätzlichem physikalisch motivierten Strafterm.

        Parameter:
        - y_target: Zielwerte
        - y_pred: Vorhersagen
        - x_input: Eingabedaten (nur notwendig für den Strafterm)

        Rückgabe:
        - Gesamtverlust (MSE + Strafterm)
        """
        criterion = nn.MSELoss()
        mse_loss = criterion(y_target.squeeze(), y_pred.squeeze())

        if x_input is not None and y_pred.requires_grad or y_target.requires_grad:
            x_input = x_input.clone().detach().requires_grad_(True)
            y_pred_physics = self(x_input)

            # dy/dx1 (x1 = Feature mit Index 0)
            dy_dx = torch.autograd.grad(
                outputs=y_pred_physics,
                inputs=x_input,
                grad_outputs=torch.ones_like(y_pred_physics),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            # ToDo: modular implementieren
            if x_input.size()[1]== 13:
                v = x_input[:, 10]              # Feature v
                a = x_input[:, 1]               # Feature a
                mrr = x_input[:, 8]             # Feature mrr
                f = x_input[:, 5]               # Feature F

                dy_da = dy_dx[:, 1]  # Ableitung nach a
                dy_dv = dy_dx[:, 10]
                dy_df = dy_dx[:, 5]
                dy_dmrr = dy_dx[:, 8]
            elif x_input.size()[1] == 4:
                v = x_input[:, 3]  # Feature v
                a = x_input[:, 0]  # Feature a
                mrr = x_input[:, 2]  # Feature mrr
                f = x_input[:, 1]  # Feature F

                dy_da = dy_dx[:, 0]  # Ableitung nach a
                dy_dv = dy_dx[:, 3]
                dy_df = dy_dx[:, 1]
                dy_dmrr = dy_dx[:, 2]
            else:
                throw_error(self, 'x_input hast the wrong size')

            constraint = (dy_da - self.c_1 * v) + (dy_dv - (self.c_1 * a + 2 * self.c_2 * torch.abs(v))) + (dy_df - self.c_3 * mrr) + (dy_dmrr - self.c_3 * f) #
            penalty = torch.mean(constraint ** 2)  # L2-Strafterm

            return mse_loss + self.penalty_weight * penalty

        return mse_loss

    def scaled_to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif hasattr(data, 'values'):
            data_scaled = self.scale_data(data.values)
            return torch.tensor(data_scaled, dtype=torch.float32).to(self.device)
        else:
            # Falls numpy array oder anderes
            data_scaled = self.scale_data(data)
            return torch.tensor(data_scaled, dtype=torch.float32).to(self.device)

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Überschreibt das Training, um x_input an die Loss-Funktion zu übergeben.
        """
        original_criterion = self.criterion

        # Patch `self.criterion` temporär für das Training
        current_input = self.scaled_to_tensor(X_train)

        def custom_train_model(*args, **kwargs):
            nonlocal current_input
            original_train_model = super(PiNN, self).train_model

            def patched_criterion(y_target, y_pred):
                return original_criterion(y_target, y_pred, x_input=current_input)

            self.criterion = patched_criterion
            result = original_train_model(*args, **kwargs)
            self.criterion = original_criterion  # Restore
            return result

        # Trick: train_model aus BaseNetModel verwendet `self.criterion`
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

class PiNNAdaptiv(Net):
    def __init__(self, *args, name="PiNN_Erd_adaptive", penalty_weight=1, optimizer_type='adam', c_1=-6.23e-6, c_2=2.27e-7, c_3=-2.7e-6,**kwargs):
        super(PiNNAdaptiv, self).__init__(*args, **kwargs)
        self.name = name
        self.penalty_weight = penalty_weight
        self.optimizer_type = optimizer_type
        self.c_1 = nn.Parameter(torch.tensor(c_1))
        self.c_2 = nn.Parameter(torch.tensor(c_2))
        self.c_3 = nn.Parameter(torch.tensor(c_3))

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

            if x_input.size()[1] == 13:
                v = x_input[:, 10]
                a = x_input[:, 1]
                mrr = x_input[:, 8]
                f = x_input[:, 5]

                dy_da = dy_dx[:, 1]
                dy_dv = dy_dx[:, 10]
                dy_df = dy_dx[:, 5]
                dy_dmrr = dy_dx[:, 8]
            elif x_input.size()[1] == 4:
                v = x_input[:, 3]
                a = x_input[:, 0]
                mrr = x_input[:, 2]
                f = x_input[:, 1]

                dy_da = dy_dx[:, 0]
                dy_dv = dy_dx[:, 3]
                dy_df = dy_dx[:, 1]
                dy_dmrr = dy_dx[:, 2]
            else:
                raise ValueError('x_input has the wrong size')

            constraint = (dy_da - self.c_1 * v) + (dy_dv - (self.c_1 * a + 2 * self.c_2 * torch.abs(v))) + (dy_df - self.c_3 * mrr) + (dy_dmrr - self.c_3 * f)
            penalty = torch.mean(constraint ** 2)

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

        current_input = self.scaled_to_tensor(X_train)

        def custom_train_model(*args, **kwargs):
            nonlocal current_input
            original_train_model = super(PiNNAdaptiv, self).train_model

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

class PiNNErdMatrix(Net):
    def __init__(self, *args, name="PiNN_Erd_Matrix", penalty_weight=1, optimizer_type='adam', theta_init=None, **kwargs):
        super(PiNNErdMatrix, self).__init__(*args, **kwargs)
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
            original_train_model = super(PiNNErdMatrix, self).train_model

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
                throw_error(self, 'x_input hat die falsche Größe')

        return mse_loss

    def get_documentation(self):
        documentation = {"hyperparameters": {
            "learning_rate": self.learning_rate,
            "n_hidden_size": self.n_hidden_size,
            "n_hidden_layers": self.n_hidden_layers,
            "n_activation_function": self.activation.__class__.__name__,
            "optimizer_type": self.optimizer_type,
            "penalty_weight": self.penalty_weight,
            "theta_init": self.theta_init.tolist(),
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


class NeuralCDE(mb.BaseNetModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=2,
                 activation=nn.ReLU, learning_rate=0.001, name="NeuralCDE", optimizer_type='adam',
                 cde_hidden_size=None, interpolation='cubic', solver='dopri5', rtol=1e-3, atol=1e-5,
                 sequence_length=10, overlap=0.5, auto_sequential=True):
        """
        Neural Controlled Differential Equation model.

        Parameters
        ----------
        input_size : int, optional
            The number of input features (channels). If None, it will be set during the first forward pass.
        output_size : int
            The number of output features.
        n_hidden_size : int, optional
            The number of features in each hidden layer. If None, it will be set to the input size during initialization.
        n_hidden_layers : int
            The number of hidden layers in the vector field network.
        activation : torch.nn.Module
            The activation function to be used in the hidden layers.
        learning_rate : float
            The learning rate for the optimizer.
        name : str
            The name of the model.
        optimizer_type : str
            The type of optimizer to use.
        cde_hidden_size : int, optional
            Hidden state size for the CDE. If None, it will be set to n_hidden_size.
        interpolation : str
            Interpolation method for the control path ('cubic', 'linear').
        solver : str
            ODE solver ('dopri5', 'rk4', 'euler').
        rtol : float
            Relative tolerance for the ODE solver.
        atol : float
            Absolute tolerance for the ODE solver.
        sequence_length : int
            Length of each sequence when converting tabular to sequential data.
        overlap : float
            Overlap ratio between consecutive sequences (0 to 1).
        auto_sequential : bool
            If True, automatically convert tabular data to sequential format.
        """
        super(NeuralCDE, self).__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type
        )

        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation()
        self.cde_hidden_size = cde_hidden_size
        self.interpolation = interpolation
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.auto_sequential = auto_sequential

        # Store original data mappings for compatibility
        self.sequence_to_original_mapping = None

        # Initialize layers only if input_size is provided
        if self.input_size is not None:
            self._initialize()

    def _initialize(self):
        """Initialize the neural network layers."""
        self.scaler = None

        # Set default sizes if not provided
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size
        if self.cde_hidden_size is None:
            self.cde_hidden_size = self.n_hidden_size

        # Initial hidden state network
        self.initial_fc1 = nn.Linear(self.input_size, self.n_hidden_size)
        self.initial_fc2 = nn.Linear(self.n_hidden_size, self.cde_hidden_size)

        # Vector field network
        # Input: hidden state + control input
        vector_field_input_size = self.cde_hidden_size + self.input_size
        self.vector_fc1 = nn.Linear(vector_field_input_size, self.n_hidden_size)
        self.vector_fcs = nn.ModuleList([
            nn.Linear(self.n_hidden_size, self.n_hidden_size)
            for _ in range(self.n_hidden_layers)
        ])
        # Output: cde_hidden_size * input_size (for matrix multiplication)
        self.vector_fc_out = nn.Linear(self.n_hidden_size, self.cde_hidden_size * self.input_size)

        # Readout network
        self.readout_fc1 = nn.Linear(self.cde_hidden_size, self.n_hidden_size)
        self.readout_fc2 = nn.Linear(self.n_hidden_size, self.output_size)

        # Move to device
        self.to(self.device)

    def prepare_sequential_data(self, X, y=None):
        """
        Convert tabular data to sequential format for Neural CDE.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data of shape (n_samples, n_features).
        y : np.ndarray or pd.Series, optional
            Target data. If provided, will be aligned with sequences.

        Returns
        -------
        tuple
            (X_sequential, y_sequential, mapping) where mapping tracks original indices.
        """
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = X

        n_samples, n_features = X_array.shape
        step_size = max(1, int(self.sequence_length * (1 - self.overlap)))

        sequences = []
        y_sequences = []
        mapping = []

        for i in range(0, n_samples - self.sequence_length + 1, step_size):
            sequences.append(X_array[i:i + self.sequence_length])
            mapping.append(list(range(i, i + self.sequence_length)))

            if y is not None:
                if hasattr(y, 'values'):
                    y_array = y.values
                elif hasattr(y, 'to_numpy'):
                    y_array = y.to_numpy()
                else:
                    y_array = y
                # Use the last target value of the sequence
                y_sequences.append(y_array[i + self.sequence_length - 1])

        X_sequential = np.array(sequences)

        if y is not None:
            y_sequential = np.array(y_sequences)
            return X_sequential, y_sequential, mapping
        else:
            return X_sequential, None, mapping

    def vector_field_func(self, t, z, X_interp):
        """
        Vector field function for the CDE.

        Parameters
        ----------
        t : torch.Tensor
            Current time point.
        z : torch.Tensor
            Current hidden state of shape (batch_size, cde_hidden_size).
        X_interp : torchcde.CubicSpline or torchcde.LinearInterpolation
            Interpolated control path.

        Returns
        -------
        torch.Tensor
            Derivative of the hidden state.
        """
        # Evaluate the control path at time t
        X_t = X_interp.evaluate(t)  # Shape: (batch_size, input_size)

        # Concatenate hidden state and control input
        z_and_X = torch.cat([z, X_t], dim=-1)  # Shape: (batch_size, cde_hidden_size + input_size)

        # Pass through vector field network
        x = self.activation(self.vector_fc1(z_and_X))
        for fc in self.vector_fcs:
            x = self.activation(fc(x))
        output = self.vector_fc_out(x)  # Shape: (batch_size, cde_hidden_size * input_size)

        # Reshape to (batch_size, cde_hidden_size, input_size)
        output = output.view(z.size(0), self.cde_hidden_size, self.input_size)

        # Compute dX/dt at time t
        dX_dt = X_interp.derivative(t)  # Shape: (batch_size, input_size)

        # Matrix multiplication: (batch_size, cde_hidden_size, input_size) @ (batch_size, input_size, 1)
        # -> (batch_size, cde_hidden_size, 1) -> (batch_size, cde_hidden_size)
        dz_dt = torch.bmm(output, dX_dt.unsqueeze(-1)).squeeze(-1)

        return dz_dt

    def forward(self, X):
        """
        Defines the forward pass of the Neural CDE.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size).

        Returns
        -------
        torch.Tensor
            The output tensor from the neural network.
        """
        # Handle case where X is 2D (tabular data)
        if len(X.shape) == 2:
            # Convert to sequential format
            batch_size, n_features = X.shape
            # Create artificial sequences by repeating the same sample
            X = X.unsqueeze(1).repeat(1, self.sequence_length, 1)

        # Handle dynamic initialization like in Net class
        if self.input_size is None:
            self.input_size = X.shape[-1]  # Last dimension is input_size for sequential data
            if self.n_hidden_size is None:
                self.n_hidden_size = self.input_size
            if self.cde_hidden_size is None:
                self.cde_hidden_size = self.n_hidden_size
            self._initialize()

        batch_size, seq_len, input_size = X.shape

        # Create time points
        t = torch.linspace(0, 1, seq_len, device=X.device, dtype=X.dtype)

        # Add time dimension to create coefficients for interpolation
        # Shape: (batch_size, seq_len, input_size + 1) where +1 is for time
        t_expanded = t.unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, 1)
        X_with_time = torch.cat([t_expanded, X], dim=-1)

        # Create interpolation
        if self.interpolation == 'cubic':
            X_interp = torchcde.CubicSpline(X_with_time)
        elif self.interpolation == 'linear':
            X_interp = torchcde.LinearInterpolation(X_with_time)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")

        # Initial hidden state network
        z0 = self.activation(self.initial_fc1(X[:, 0]))  # Use first time point
        z0 = self.initial_fc2(z0)

        # Solve the CDE
        z_T = torchcde.cdeint(
            X=X_interp,
            func=self.vector_field_func,
            z0=z0,
            t=torch.tensor([0., 1.], device=X.device, dtype=X.dtype),
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )

        # Take the final state (at time T=1)
        z_final = z_T[-1]  # Shape: (batch_size, cde_hidden_size)

        # Apply readout network
        x = self.activation(self.readout_fc1(z_final))
        x = self.readout_fc2(x)

        return x

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Override train_model to handle tabular to sequential conversion.
        """
        if self.auto_sequential and (len(X_train.shape) == 2 if isinstance(X_train, np.ndarray) else
        not isinstance(X_train, list)):
            # Convert tabular data to sequential
            X_train_seq, y_train_seq, _ = self.prepare_sequential_data(X_train, y_train)
            X_val_seq, y_val_seq, _ = self.prepare_sequential_data(X_val, y_val)

            # Call parent train_model with sequential data
            return super().train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, **kwargs)
        else:
            # Data is already in correct format or is batched
            return super().train_model(X_train, y_train, X_val, y_val, **kwargs)

    def test_model(self, X, y_target, criterion_test=None):
        """
        Override test_model to handle tabular to sequential conversion.
        """
        if self.auto_sequential and len(X.shape) == 2:
            # Convert tabular data to sequential
            X_seq, y_seq, mapping = self.prepare_sequential_data(X, y_target)
            self.sequence_to_original_mapping = mapping

            # Call parent test_model with sequential data
            loss, y_pred_seq = super().test_model(X_seq, y_seq, criterion_test)

            # Map predictions back to original data size
            y_pred_full = np.full(len(y_target), np.nan)

            for i, seq_mapping in enumerate(mapping):
                # Use the prediction for the last element of each sequence
                original_idx = seq_mapping[-1]
                if i < len(y_pred_seq):
                    y_pred_full[original_idx] = y_pred_seq[i]

            # Fill any remaining NaN values with the mean of available predictions
            nan_mask = np.isnan(y_pred_full)
            if np.any(nan_mask):
                mean_pred = np.nanmean(y_pred_full)
                y_pred_full[nan_mask] = mean_pred

            return loss, y_pred_full
        else:
            # Data is already in correct format
            return super().test_model(X, y_target, criterion_test)

    def predict(self, X):
        """
        Override predict to handle tabular to sequential conversion.
        """
        if self.auto_sequential and len(X.shape) == 2:
            X_seq, _, mapping = self.prepare_sequential_data(X)
            X_tensor = self.scaled_to_tensor(X_seq)
            y_pred_seq = self(X_tensor).detach().cpu().numpy()

            # Map predictions back to original data size
            y_pred_full = np.full(X.shape[0], np.nan)

            for i, seq_mapping in enumerate(mapping):
                original_idx = seq_mapping[-1]
                if i < len(y_pred_seq):
                    y_pred_full[original_idx] = y_pred_seq[i].item() if y_pred_seq[i].ndim == 0 else y_pred_seq[i][0]

            # Fill any remaining NaN values
            nan_mask = np.isnan(y_pred_full)
            if np.any(nan_mask):
                mean_pred = np.nanmean(y_pred_full)
                y_pred_full[nan_mask] = mean_pred

            return torch.tensor(y_pred_full, device=self.device)
        else:
            return super().predict(X)

    def get_documentation(self):
        """Get model documentation including hyperparameters."""
        documentation = {
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "n_hidden_size": self.n_hidden_size,
                "n_hidden_layers": self.n_hidden_layers,
                "n_activation_function": self.activation.__class__.__name__,
                "cde_hidden_size": self.cde_hidden_size,
                "interpolation": self.interpolation,
                "solver": self.solver,
                "rtol": self.rtol,
                "atol": self.atol,
                "sequence_length": self.sequence_length,
                "overlap": self.overlap,
                "auto_sequential": self.auto_sequential,
            }
        }
        return documentation