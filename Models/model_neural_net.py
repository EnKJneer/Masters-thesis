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
import torch.optim as optim
import torch.jit
import torch.nn as nn
from matplotlib import pyplot as plt
from numpy.f2py.auxfuncs import throw_error
from sklearn.preprocessing import QuantileTransformer
from torch.xpu import device

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
                 learning_rate=0.001, name="Neural_Net", optimizer_type='adam', dropout_rate=0.0):
        """
        Initializes a configurable neural network with optional dropout for regularization.

        Parameters
        ----------
        input_size : int, optional
            The number of input features. If None, it will be set during the first training call.
        output_size : int, optional
            The number of output features. Default is 1.
        n_hidden_size : int, optional
            The number of features in each hidden layer. If None, it will be set to the input size during the first training call.
        n_hidden_layers : int, optional
            The number of hidden layers in the network. Default is 1.
        activation : str, optional
            The activation function to be used in the hidden layers. Default is 'ReLU'.
        learning_rate : float, optional
            The learning rate for the optimizer. Default is 0.001.
        name : str, optional
            The name of the model. Default is "Neural_Net".
        optimizer_type : str, optional
            The type of optimizer to use. Default is 'adam'.
        dropout_rate : float, optional
            The dropout rate for regularization. If 0.0, dropout is not applied. Default is 0.0.
        """
        super(Net, self).__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type
        )
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation = self.activation_map[activation]()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        if self.input_size is not None:
            self._initialize()

    def reset_hyperparameter(self, n_hidden_size, n_hidden_layers, learning_rate, activation, optimizer_type, dropout_rate=0.0):
        """
        Resets the hyperparameters of the neural network and reinitializes the model.

        Parameters
        ----------
        n_hidden_size : int
            The number of features in each hidden layer.
        n_hidden_layers : int
            The number of hidden layers in the network.
        learning_rate : float
            The learning rate for the optimizer.
        activation : str
            The activation function to be used in the hidden layers.
        optimizer_type : str
            The type of optimizer to use.
        dropout_rate : float, optional
            The dropout rate for regularization. If 0.0, dropout is not applied. Default is 0.0.
        """
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate
        self.activation = self.activation_map[activation]()
        self.optimizer_type = optimizer_type
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def _initialize(self):
        """
        Initializes the layers of the neural network.
        This method is called automatically during the first forward pass if the input size is not set.
        """
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
        torch.Tensor
            The output tensor from the neural network.
        """
        if self.input_size is None:
            self.input_size = x.shape[1]
            if self.n_hidden_size is None:
                self.n_hidden_size = self.input_size
            self._initialize()
        x = self.activation(self.fc1(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        for fc in self.fcs:
            x = self.activation(fc(x))
            if self.dropout_rate > 0:
                x = self.dropout(x)
        x = self.fc3(x)
        return x

    @staticmethod
    def get_reference_model(input_size=None):
        """
        Returns a reference neural network model with default parameters.

        Parameters
        ----------
        input_size : int, optional
            The number of input features for the neural network. If None, the input size will be set at runtime.

        Returns
        -------
        Net
            An instance of the Net class configured with default parameters.
        """
        return Net(input_size, 1, input_size, dropout_rate=0.0)

class RNN(mb.BaseTorchModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=1, activation='ReLU',
                 learning_rate=0.001, name="Recurrent_Neural_Net", batched_input=False, optimizer_type='adam', dropout_rate=0.0):
        """
        Initializes a configurable recurrent neural network with optional dropout for regularization.

        Parameters
        ----------
        input_size : int, optional
            The number of input features. If None, it will be set during the first training call.
        output_size : int, optional
            The number of output features. Default is 1.
        n_hidden_size : int, optional
            The number of features in each hidden layer. If None, it will be set to the input size during the first training call.
        n_hidden_layers : int, optional
            The number of hidden layers in the network. Default is 1.
        activation : str, optional
            The activation function to be used in the hidden layers. Default is 'ReLU'.
        learning_rate : float, optional
            The learning rate for the optimizer. Default is 0.001.
        name : str, optional
            The name of the model. Default is "Recurrent_Neural_Net".
        batched_input : bool, optional
            Indicates whether the input is batched (3D) or not (2D). Default is False.
        optimizer_type : str, optional
            The type of optimizer to use. Default is 'adam'.
        dropout_rate : float, optional
            The dropout rate for regularization. If 0.0, dropout is not applied. Default is 0.0.
        """
        super(RNN, self).__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type
        )
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation_name = activation
        self.activation = self.activation_map[activation]()
        self.learning_rate = learning_rate
        self.name = name
        self.batched_input = batched_input
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        if self.input_size is not None:
            self._initialize()

    def reset_hyperparameter(self, **kwargs):
        """
        Resets the hyperparameters of the recurrent neural network and reinitializes the model.
        Only updates parameters that are explicitly provided in kwargs.
        """
        # Update only if parameter exists in kwargs
        if 'n_hidden_size' in kwargs:
            self.n_hidden_size = kwargs['n_hidden_size']

        if 'n_hidden_layers' in kwargs:
            self.n_hidden_layers = kwargs['n_hidden_layers']

        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']

        if 'activation' in kwargs:
            self.activation_name = kwargs['activation']
            self.activation = self.activation_map[self.activation_name]()

        if 'optimizer_type' in kwargs:
            self.optimizer_type = kwargs['optimizer_type']

        if 'dropout_rate' in kwargs:
            self.dropout_rate = kwargs['dropout_rate']
            self.dropout = nn.Dropout(self.dropout_rate)

        # Reinitialize only if input_size exists and at least one parameter was updated
        if self.input_size is not None and any(param in kwargs for param in [
            'n_hidden_size', 'n_hidden_layers', 'activation', 'dropout_rate'
        ]):
            self._initialize()

    def _initialize(self):
        """Initializes the layers of the recurrent neural network."""
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size
        self.rnn = nn.RNN(
            self.input_size,
            self.n_hidden_size,
            self.n_hidden_layers,
            batch_first=self.batched_input,
            dropout=self.dropout_rate if self.n_hidden_layers > 1 else 0.0  # Dropout only applies between RNN layers
        )
        self.fc = nn.Linear(self.n_hidden_size, self.output_size)
        self.to(self.device)

    def forward(self, x):
        """
        Defines the forward pass of the recurrent neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the recurrent neural network.

        Returns
        -------
        torch.Tensor
            The output tensor from the recurrent neural network.
        """
        if self.input_size is None:
            self.input_size = x.shape[-1]
            if self.n_hidden_size is None:
                self.n_hidden_size = self.input_size
            self._initialize()
            self.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add batch dimension
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
        Returns a reference recurrent neural network model with default parameters.

        Parameters
        ----------
        input_size : int, optional
            The number of input features for the neural network. If None, the input size will be set at runtime.

        Returns
        -------
        RNN
            An instance of the RNN class configured with default parameters.
        """
        return RNN(input_size, 1, input_size, dropout_rate=0.0)

class RNN_noTime(mb.BaseTorchModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=1, activation='ReLU',
                 learning_rate=0.001, name="Recurrent_Neural_Net", batched_input=False, optimizer_type='adam', dropout_rate=0.0):
        """
        Initializes a configurable recurrent neural network with optional dropout for regularization.

        Parameters
        ----------
        input_size : int, optional
            The number of input features. If None, it will be set during the first training call.
        output_size : int, optional
            The number of output features. Default is 1.
        n_hidden_size : int, optional
            The number of features in each hidden layer. If None, it will be set to the input size during the first training call.
        n_hidden_layers : int, optional
            The number of hidden layers in the network. Default is 1.
        activation : str, optional
            The activation function to be used in the hidden layers. Default is 'ReLU'.
        learning_rate : float, optional
            The learning rate for the optimizer. Default is 0.001.
        name : str, optional
            The name of the model. Default is "Recurrent_Neural_Net".
        batched_input : bool, optional
            Indicates whether the input is batched (3D) or not (2D). Default is False.
        optimizer_type : str, optional
            The type of optimizer to use. Default is 'adam'.
        dropout_rate : float, optional
            The dropout rate for regularization. If 0.0, dropout is not applied. Default is 0.0.
        """
        super(RNN_noTime, self).__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type
        )
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation_name = activation
        self.activation = self.activation_map[activation]()
        self.learning_rate = learning_rate
        self.name = name
        self.batched_input = batched_input
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        if self.input_size is not None:
            self._initialize()

    def reset_hyperparameter(self, **kwargs):
        """
        Resets the hyperparameters of the recurrent neural network and reinitializes the model.
        Only updates parameters that are explicitly provided in kwargs.
        """
        # Update only if parameter exists in kwargs
        if 'n_hidden_size' in kwargs:
            self.n_hidden_size = kwargs['n_hidden_size']

        if 'n_hidden_layers' in kwargs:
            self.n_hidden_layers = kwargs['n_hidden_layers']

        if 'learning_rate' in kwargs:
            self.learning_rate = kwargs['learning_rate']

        if 'activation' in kwargs:
            self.activation_name = kwargs['activation']
            self.activation = self.activation_map[self.activation_name]()

        if 'optimizer_type' in kwargs:
            self.optimizer_type = kwargs['optimizer_type']

        if 'dropout_rate' in kwargs:
            self.dropout_rate = kwargs['dropout_rate']
            self.dropout = nn.Dropout(self.dropout_rate)

        # Reinitialize only if input_size exists and at least one parameter was updated
        if self.input_size is not None and any(param in kwargs for param in [
            'n_hidden_size', 'n_hidden_layers', 'activation', 'dropout_rate'
        ]):
            self._initialize()

    def _initialize(self):
        """Initializes the layers of the recurrent neural network."""
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size
        self.rnn = nn.RNN(
            self.input_size,
            self.n_hidden_size,
            self.n_hidden_layers,
            batch_first=self.batched_input,
            dropout=self.dropout_rate if self.n_hidden_layers > 1 else 0.0  # Dropout only applies between RNN layers
        )
        self.fc = nn.Linear(self.n_hidden_size, self.output_size)
        self.to(self.device)

    def forward(self, x):
        """
        Defines the forward pass of the recurrent neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the recurrent neural network.

        Returns
        -------
        torch.Tensor
            The output tensor from the recurrent neural network.
        """
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
        Returns a reference recurrent neural network model with default parameters.

        Parameters
        ----------
        input_size : int, optional
            The number of input features for the neural network. If None, the input size will be set at runtime.

        Returns
        -------
        RNN
            An instance of the RNN class configured with default parameters.
        """
        return RNN(input_size, 1, input_size, dropout_rate=0.0)


class LSTM(mb.BaseTorchModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=1, activation='ReLU',
                 learning_rate=0.001, name="LSTM", batched_input=False, optimizer_type='adam', dropout_rate=0.0):
        """
        Initializes a configurable Long Short-Term Memory (LSTM) network with optional dropout for regularization.

        Parameters
        ----------
        input_size : int, optional
            The number of input features. If None, it will be set during the first training call.
        output_size : int, optional
            The number of output features. Default is 1.
        n_hidden_size : int, optional
            The number of features in each hidden layer. If None, it will be set to the input size during the first training call.
        n_hidden_layers : int, optional
            The number of hidden layers in the network. Default is 1.
        activation : str, optional
            The activation function to be used in the hidden layers. Default is 'ReLU'.
        learning_rate : float, optional
            The learning rate for the optimizer. Default is 0.001.
        name : str, optional
            The name of the model. Default is "LSTM".
        batched_input : bool, optional
            Indicates whether the input is batched (3D) or not (2D). Default is False.
        optimizer_type : str, optional
            The type of optimizer to use. Default is 'adam'.
        dropout_rate : float, optional
            The dropout rate for regularization. If 0.0, dropout is not applied. Default is 0.0.
        """
        super(LSTM, self).__init__(
            input_size=input_size,
            output_size=output_size,
            name=name,
            learning_rate=learning_rate,
            optimizer_type=optimizer_type
        )
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.activation_name = activation
        self.activation = self.activation_map[activation]()
        self.learning_rate = learning_rate
        self.name = name
        self.batched_input = batched_input
        self.dropout_rate = dropout_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        if self.input_size is not None:
            self._initialize()

    def reset_hyperparameter(self, n_hidden_size, n_hidden_layers, learning_rate, activation, optimizer_type, dropout_rate=0.0):
        """
        Resets the hyperparameters of the LSTM network and reinitializes the model.

        Parameters
        ----------
        n_hidden_size : int
            The number of features in each hidden layer.
        n_hidden_layers : int
            The number of hidden layers in the network.
        learning_rate : float
            The learning rate for the optimizer.
        activation : str
            The activation function to be used in the hidden layers.
        optimizer_type : str
            The type of optimizer to use.
        dropout_rate : float, optional
            The dropout rate for regularization. If 0.0, dropout is not applied. Default is 0.0.
        """
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.learning_rate = learning_rate
        self.activation_name = activation
        self.activation = self.activation_map[activation]()
        self.optimizer_type = optimizer_type
        self.dropout_rate = dropout_rate
        if self.input_size is not None:
            self._initialize()

    def _initialize(self):
        """Initializes the layers of the LSTM network."""
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size
        self.rnn = nn.LSTM(
            self.input_size,
            self.n_hidden_size,
            self.n_hidden_layers,
            batch_first=self.batched_input,
            dropout=self.dropout_rate if self.n_hidden_layers > 1 else 0.0  # Dropout only applies between LSTM layers
        )
        self.fc = nn.Linear(self.n_hidden_size, self.output_size)
        self.to(self.device)

    def forward(self, x):
        """
        Defines the forward pass of the LSTM network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape [batch_size, seq_len, input_size] or [seq_len, input_size].

        Returns
        -------
        torch.Tensor
            Output tensor from the LSTM network.
        """
        if self.input_size is None:
            self.input_size = x.shape[-1]
            if self.n_hidden_size is None:
                self.n_hidden_size = self.input_size
            self._initialize()
            self.to(self.device)
        if x.dim() == 2:  # [seq_len, input_size]
            x = x.unsqueeze(1)  # → [seq_len, 1, input_size]
        x = x.to(self.device)
        x = x.contiguous()  # Important for cuDNN
        batch_size = x.size(1)
        h0 = torch.zeros(self.n_hidden_layers, batch_size, self.n_hidden_size, device=self.device)
        c0 = torch.zeros(self.n_hidden_layers, batch_size, self.n_hidden_size, device=self.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out)
        return out

    @staticmethod
    def get_reference_model(input_size=None):
        """
        Returns a reference LSTM model with default parameters.

        Parameters
        ----------
        input_size : int, optional
            The number of input features for the LSTM network. If None, the input size will be set at runtime.

        Returns
        -------
        LSTM
            An instance of the LSTM class configured with default parameters.
        """
        return LSTM(input_size, 1, input_size, dropout_rate=0.0)

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

class PiNNErd(RNN):
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
        super(PiNNErd, self)._initialize()

        if self.theta_init is not None:
            self.theta = nn.Parameter(torch.tensor(self.theta_init, dtype=torch.float32))
        else:
            self.theta = nn.Parameter(torch.zeros(4, 5, dtype=torch.float32))  # Achsen: x, y, z, sp
        self.to(self.device)

    def reset_hyperparameter(self, **kwargs):
        """
        Resets the hyperparameters of the recurrent neural network and reinitializes the model.

        """
        kwargs = kwargs.copy()  # oder dict(kwargs)
        if 'penalty_weight' in kwargs:
            self.penalty_weight = kwargs.pop('penalty_weight')
        if kwargs:
            super().reset_hyperparameter(**kwargs)

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        """
        Überschreibt das Training, um x_input an die Loss-Funktion zu übergeben.
        """
        original_criterion = self.criterion

        # Patch `self.criterion` temporär für das Training
        current_input = self.scaled_to_tensor(X_train)

        def custom_train_model(*args, **kwargs):
            nonlocal current_input
            original_train_model = super(PiNNThermo, self).train_model

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
                            a[:, i] * self.theta[i, 0] + # Teil von dI/dv
                            v[:, i] * self.theta[i, 0] + # dI/da
                            v[:, i] * self.theta[i, 1] * torch.sign(v[:, i]) + # Teil von dI/dv
                            f[:, i] * self.theta[i, 2] +  # dI/dMRR
                            mrr.squeeze(1) * self.theta[i, 2]  # dI/dF
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
                        v * self.theta[1, 0] +
                        v * self.theta[1, 1] * torch.sign(v)  +
                        f * self.theta[1, 2] +
                        mrr * self.theta[1, 2]
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

class PiNNThermo(RNN):
    def __init__(self, *args, name="PiNN_Erd", penalty_weight=1, optimizer_type='adam', theta_init=None, **kwargs):
        super(PiNNThermo, self).__init__(*args, **kwargs)
        self.name = name
        self.penalty_weight = penalty_weight
        self.optimizer_type = optimizer_type
        self.theta_init = theta_init
        self.epsilon = 9e-1

        # Neue Parameter-Matrix für alle Achsen
        if theta_init is not None:
            if theta_init.shape != (4, 5):
                raise ValueError("theta_init must have shape (4, 5)")
            self.theta = nn.Parameter(torch.tensor(theta_init, dtype=torch.float32))
        else:
            self.theta = nn.Parameter(torch.zeros(4, 5, dtype=torch.float32))  # Achsen: x, y, z, sp

    def _initialize(self):
        """Initialize the layers of the neural network."""
        super(PiNNThermo, self)._initialize()

        if self.theta_init is not None:
            self.theta = nn.Parameter(torch.tensor(self.theta_init, dtype=torch.float32))
        else:
            self.theta = nn.Parameter(torch.zeros(4, 5, dtype=torch.float32))  # Achsen: x, y, z, sp
        self.to(self.device)


    def forward(self, x):
        """
        Defines the forward pass of the recurrent neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the recurrent neural network.

        Returns
        -------
        torch.Tensor
            The output tensor from the recurrent neural network.
        """
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
        batch_size = x.size(1) # keine zeitiche rückopplung
        h0 = torch.zeros(self.n_hidden_layers, batch_size, self.n_hidden_size, device=self.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

    def reset_hyperparameter(self, **kwargs):
        """
        Resets the hyperparameters of the recurrent neural network and reinitializes the model.

        """
        kwargs = kwargs.copy()  # oder dict(kwargs)
        if 'penalty_weight' in kwargs:
            self.penalty_weight = kwargs.pop('penalty_weight')
        if kwargs:
            super().reset_hyperparameter(**kwargs)

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=100, draw_loss=False, epsilon=0.00005,
                    trial=None, n_outlier=12, reset_parameters=True, patience_stop=10, patience_lr=3, **kwargs):

        draw_loss = False

        # Prüfen, ob Inputs Listen sind
        is_batched_train = isinstance(X_train, list) and isinstance(y_train, list)
        is_batched_val = isinstance(X_val, list) and isinstance(y_val, list)

        assert (not is_batched_train) or (len(X_train) == len(y_train)), "Trainingslist must have the same length"
        assert (not is_batched_val) or (len(X_val) == len(y_val)), "Validierungslisten must have the same length"

        print(f"Device: {self.device} | Batched: {is_batched_train}")

        """Needed for initialization of the layer."""
        flag_initialization = False
        if self.input_size is None:
            # Define input size and set flag for initialization
            if is_batched_train:
                self.input_size = X_train[0].shape[1]
            else:
                self.input_size = X_train.shape[1]
            flag_initialization = True
        if flag_initialization or reset_parameters: #
            self._initialize()

        if self.optimizer_type.lower() == 'adam' or self.optimizer_type.lower() == 'sgd':
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type.lower() == 'quasi_newton':
            optimizer = optim.LBFGS(self.parameters(), lr=self.learning_rate, max_iter=20, history_size=10)
        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_lr)

        if draw_loss:
            plt.ion()  # Interaktiven Modus aktivieren

            loss_vals, epochs, loss_train = [], [], []
            fig, ax = plt.subplots()
            line_val, = ax.plot(epochs, loss_vals, 'r-', label='validation')
            line_train, = ax.plot(epochs, loss_train, 'b-', label='training')
            ax.legend()

        best_val_error = float('inf')
        best_model_state = self.state_dict()

        patience_counter = 0



        for epoch in range(n_epochs):
            self.train()

            train_losses = []

            def closure():
                optimizer.zero_grad()
                if is_batched_train:
                    loss_total = 0
                    for batch_x, batch_y in zip(X_train, y_train):
                        batch_x_tensor = self.scaled_to_tensor(batch_x)
                        batch_y_tensor = self.to_tensor(batch_y)
                        output = self(batch_x_tensor)
                        loss = self.criterion(batch_y_tensor, output, batch_x_tensor)
                        loss.backward()
                        loss_total += loss
                    return loss_total
                else:
                    x_tensor = self.scaled_to_tensor(X_train)
                    y_tensor = self.to_tensor(y_train)
                    output = self(x_tensor)
                    loss = self.criterion(y_tensor, output, x_tensor)
                    loss.backward()
                    return loss

            if self.optimizer_type.lower() == 'quasi_newton':
                loss = optimizer.step(closure)
                train_losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
            else:
                if not is_batched_train:
                    loss = closure()
                    optimizer.step()
                    train_losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
                else:
                    for batch_x, batch_y in zip(X_train, y_train):
                        optimizer.zero_grad()
                        batch_x_tensor = self.scaled_to_tensor(batch_x)
                        batch_y_tensor = self.to_tensor(batch_y)
                        output = self(batch_x_tensor)
                        loss = self.criterion(output, batch_y_tensor, batch_x_tensor)
                        loss.backward()
                        optimizer.step()
                        train_losses.append(loss.item())

            self.eval()
            val_losses = []
            with torch.no_grad():
                if is_batched_val:
                    for batch_x, batch_y in zip(X_val, y_val):
                        batch_x_tensor = self.scaled_to_tensor(batch_x)
                        batch_y_tensor = self.to_tensor(batch_y)

                        output = self(batch_x_tensor)
                        val_loss = self.criterion(batch_y_tensor, output, batch_x_tensor)
                        val_losses.append(val_loss.item())
                else:
                    x_tensor = self.scaled_to_tensor(X_val)
                    y_tensor = self.to_tensor(y_val)

                    output = self(x_tensor)
                    val_loss = self.criterion(y_tensor, output, x_tensor)
                    val_losses.append(val_loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)

            if avg_val_loss < best_val_error - epsilon:
                best_val_error = avg_val_loss
                best_model_state = self.state_dict()
                patience_counter = 0
            elif epoch > (n_epochs / 100):
                patience_counter += 1

            scheduler.step(avg_val_loss)

            if patience_counter >= patience_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if trial is not None:
                trial.report(avg_val_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if draw_loss:
                epochs.append(epoch)
                loss_vals.append(avg_val_loss)
                loss_train.append(loss.item())

                if draw_loss:
                    epochs.append(epoch)
                    loss_vals.append(avg_val_loss)
                    loss_train.append(loss.item())
                    if epoch == 1 or epoch % 5 == 0 or epoch == n_epochs - 1:  # Nur alle 5 Epochen oder am Ende aktualisieren
                        line_val.set_xdata(epochs)
                        line_val.set_ydata(loss_vals)
                        line_train.set_xdata(epochs)
                        line_train.set_ydata(loss_train)
                        ax.relim()
                        ax.autoscale_view()
                        plt.draw()
                        plt.pause(0.001)  # Kurze Pause, um das Fenster zu aktualisieren

            print(f'{self.name}: Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.4f} Val Error: {avg_val_loss:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        if draw_loss:
            plt.show()
            plt.pause(1)  # 1 Sekunden warten
            plt.close(fig)  # Plot explizit schließen

        self.load_state_dict(best_model_state)

        return best_val_error

    def criterion(self, y_target, y_pred, x_input=None):
        criterion = nn.MSELoss()
        mse_loss = criterion(y_target.squeeze(), y_pred.squeeze())

        if x_input is not None and (y_pred.requires_grad or y_target.requires_grad):
            # 1. x_input auf GPU und requires_grad setzen
            if x_input.device != self.device:
                x_input = x_input.to(self.device)

            # WICHTIG: Detach first, dann requires_grad
            x_input = x_input.detach().requires_grad_(True)

            # 2. Stelle sicher, dass Modell im Training-Modus ist
            was_training = self.training
            self.train()

            # 3. Forward-Pass mit disabled CuDNN
            with torch.backends.cudnn.flags(enabled=False):
                y_pred_physics = self(x_input)

            # 4. Zurück zum ursprünglichen Modus
            if not was_training:
                self.eval()

            # 5. Gradient berechnen
            dy_dx = torch.autograd.grad(
                outputs=y_pred_physics,
                inputs=x_input,
                grad_outputs=torch.ones_like(y_pred_physics),
                create_graph=True,
                retain_graph=True,
            )[0]

            if x_input.size(1) == 13:
                f = x_input[:, 4:8]
                mrr = x_input[:, 8:9]
                v = x_input[:, 9:13]

                dy_da = dy_dx[:, 0:4]
                dy_df = dy_dx[:, 4:8]
                dy_dmrr = dy_dx[:, 8:9]
                dy_dv = dy_dx[:, 9:13]

                # Safe v
                v_safe = torch.where(torch.abs(v) < self.epsilon, torch.sign(v) * self.epsilon, v)

                # Theta broadcasting
                theta_0 = self.theta[:, 0].view(1, 4)
                theta_1 = self.theta[:, 1].view(1, 4)
                theta_2 = self.theta[:, 2].view(1, 4)

                # Maske für sp-Achse
                is_sp = torch.tensor([True, False, False, False], device=self.device).view(1, 4)

                # Terms berechnen
                dv_term_sp = (dy_dv - theta_0 - theta_2 * f * mrr / v_safe ** 2) ** 2
                dv_term_other = (dy_dv - theta_0) ** 2
                dv_term = torch.where(is_sp, dv_term_sp, dv_term_other)

                df_term_sp = (dy_df - theta_2 * f / v_safe) ** 2
                df_term_other = (dy_df - theta_2) ** 2
                df_term = torch.where(is_sp, df_term_sp, df_term_other)

                dmrr_term_sp = (dy_dmrr - theta_2 * mrr / v_safe) ** 2
                dmrr_term_other = (dy_dmrr - theta_2) ** 2
                dmrr_term = torch.where(is_sp, dmrr_term_sp, dmrr_term_other)

                da_term = (dy_da - theta_1) ** 2

                # Constraint
                constraint = dv_term + da_term + df_term + dmrr_term
                penalty = constraint.mean()

                return mse_loss + self.penalty_weight * penalty
            else:
                raise ValueError('x_input hat die falsche Größe')

        return mse_loss

    def criterion_old(self, y_target, y_pred, x_input=None):
        criterion = nn.MSELoss()
        mse_loss = criterion(y_target.squeeze(), y_pred.squeeze())

        if x_input is not None and (y_pred.requires_grad or y_target.requires_grad):
            # 1. CUDA-Kontext vorab initialisieren
            torch.zeros(1, device=self.device)

            # 2. x_input auf GPU verschieben UND Gradienten aktivieren
            x_input = x_input.clone().detach().requires_grad_(True).to(self.device)

            # 3. Modellvorhersage (mit deaktiviertem cuDNN)
            with torch.backends.cudnn.flags(enabled=False):
                y_pred_physics = self(x_input)

            # 4. Gradientenberechnung (ALLES muss auf GPU sein!)
            dy_dx = torch.autograd.grad(
                outputs=y_pred_physics,
                inputs=x_input,
                grad_outputs=torch.ones_like(y_pred_physics, device=self.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            if x_input.size(1) == 13:
                a = x_input[:, 0:4]  # (N, 4)
                f = x_input[:, 4:8]  # (N, 4)
                mrr = x_input[:, 8]  # (N,)
                v = x_input[:, 9:13]  # (N, 4)
                ones = torch.ones_like(mrr)  # (N,)

                dy_da = dy_dx[:, 0:4]  # (N, 4)
                dy_dv = dy_dx[:, 9:13]  # (N, 4)
                dy_df = dy_dx[:, 4:8]  # (N, 4)
                dy_dmrr = dy_dx[:, 8]  # (N,)

                constraint = []
                for i in range(4):  # 4 Achsen: sp, x, y, z

                    if i == 0:  # Sonderfall für sp-Achse (MRR-Term)
                        # Ersetze kleine Werte um überanpassung an Außreisern zu verhindern
                        v_safe = torch.where(
                            torch.abs(v[:, i]) < self.epsilon,
                            torch.sign(v[:, i]) * self.epsilon,  # Erhalte Vorzeichen, setze Betrag auf 0.9
                            v[:, i]
                        )
                        # Elementweise Berechnung der influences (skalar pro Probe)
                        constraint_i = (
                            (dy_dv[:, i] - self.theta[i, 0] * ones - self.theta[i, 2] * f[:, i] * mrr / v_safe**2) ** 2 +  # dI/dv (skalar)
                            (dy_da[:, i] - self.theta[i, 1] * ones) ** 2 +  # dI/da (skalar)
                            (dy_dmrr - self.theta[i, 2] * mrr / v_safe) ** 2 +  # dI/dF (N,)
                            (dy_df[:, i]  - self.theta[i, 2] * f[:, i]/ v_safe) ** 2 # dI/dMRR (N,)
                        )
                    else:
                        constraint_i = (
                            (dy_dv[:, i] - self.theta[i, 0] * ones) ** 2 +  # dI/dv (skalar)
                            (dy_da[:, i]  - self.theta[i, 1] * ones) ** 2 +  # dI/da (skalar)
                            (dy_df[:, i] - self.theta[i, 2] * ones) ** 2  + # dI/dF (skalar)
                            (dy_dmrr - self.theta[i, 2] * ones) ** 2 # dI/dMRR (skalar)
                        )
                    constraint.append(constraint_i.unsqueeze(1))  # (N, 1)

                constraint = torch.cat(constraint, dim=1)  # (N, 4)
                penalty = torch.mean(constraint)

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

class PiRNNFriction(RNN):
    def __init__(self, *args, name="PiRNNFriction", penalty_weight=1, optimizer_type='adam', learning_rate=0.001, theta_init=None,**kwargs):
        super(PiRNNFriction, self).__init__(*args, learning_rate=learning_rate, **kwargs)
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

        if x_input is not None and (y_pred.requires_grad or y_target.requires_grad):
            x_input = x_input.clone().detach().requires_grad_(True)

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

            constraint = []
            deriv = dy_dv + dy_df + dy_da + dy_dz
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
            constraint = torch.cat(constraint, dim=1)
            penalty = torch.mean(constraint ** 2) + 1 / 10 * torch.mean(dy_remaining ** 2)

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
            original_train_model = super(PiRNNFriction, self).train_model

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

class PhysicsInformedRNN(mb.BaseTorchModel):
    def __init__(
        self,
        input_size=None,
        output_size=1,
        n_hidden_size=None,
        n_hidden_layers=1,
        activation='ReLU',
        learning_rate=0.001,
        name="PhysicsInformedRNN",
        batched_input=False,
        optimizer_type='adam',
        dropout_rate=0.0,
        z_dim=1,
        dt=1.0,
        physics_weight=1.0,
        axis='x',  # Standardachse für a, v, F
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
        self.physics_weight = physics_weight
        self.axis = axis

        # Lerntbare Koeffizienten für die physikalische Gleichung
        self.c1 = nn.Parameter(torch.tensor(0.1))
        self.c2 = nn.Parameter(torch.tensor(0.1))
        self.c3 = nn.Parameter(torch.tensor(0.1))
        self.c4 = nn.Parameter(torch.tensor(0.1))
        self.c5 = nn.Parameter(torch.tensor(0.1))

        # Gespeicherte Indizes für a, v, F (werden in train_model/test_model gesetzt)
        self.idx_v = None
        self.idx_a = None
        self.idx_F = None

        # Initialisierung der Schichten
        if self.input_size is not None:
            self._initialize()

    def _initialize(self):
        """Initialisiert RNN, z_layer und fc."""
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size
        self.rnn = nn.RNN(
            self.input_size,
            self.n_hidden_size,
            self.n_hidden_layers,
            batch_first=self.batched_input,
            dropout=self.dropout_rate if self.n_hidden_layers > 1 else 0.0,
        )
        self.z_layer = nn.Linear(self.n_hidden_size, self.z_dim)
        self.fc = nn.Linear(self.n_hidden_size, self.output_size)
        self.to(self.device)

    def _set_physics_indices(self, X, target):
        """
        Bestimmt die Indizes von a, v, F im DataFrame X basierend auf target.
        Speichert die Indizes in self.idx_v, self.idx_a, self.idx_F.
        """
        axis = target.replace('curr_', '')
        # Extrahiere die Spaltennamen (für spätere Verwendung in _extract_physics_terms)
        self.col_v = f'v_{axis}_1_current'
        self.col_a = f'a_{axis}_1_current'
        self.col_F = f'f_{axis}_sim_1_current'

        # Bestimme die Indizes der Spalten in X
        if isinstance(X, pd.DataFrame):
            self.idx_v = X.columns.get_loc(self.col_v)
            self.idx_a = X.columns.get_loc(self.col_a)
            self.idx_F = X.columns.get_loc(self.col_F)
        else:
            raise ValueError("X muss ein DataFrame sein, um die Indizes zu bestimmen!")

    def _extract_physics_terms(self, x):
        """
        Extrahiere a, v, F aus dem Input-Tensor x basierend auf den gespeicherten Indizes.
        Unterstützt Tensoren und DataFrames.
        """
        if isinstance(x, torch.Tensor):
            # Falls x ein Tensor ist, verwende die gespeicherten Indizes
            if self.idx_v is None or self.idx_a is None or self.idx_F is None:
                raise ValueError("Indizes für a, v, F wurden nicht gesetzt! Rufe zuerst train_model/test_model auf.")
            v = x[:, :, self.idx_v].unsqueeze(-1)  # [batch, seq_len, 1]
            a = x[:, :, self.idx_a].unsqueeze(-1)
            F = x[:, :, self.idx_F].unsqueeze(-1)
            return a, v, F
        elif isinstance(x, pd.DataFrame):
            # Falls x ein DataFrame ist, verwende die Spaltennamen
            v = torch.tensor(
                x[self.col_v].values,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(-1)
            a = torch.tensor(
                x[self.col_a].values,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(-1)
            F = torch.tensor(
                x[self.col_F].values,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(-1)
            return a, v, F
        else:
            raise ValueError("Input x muss ein Tensor oder DataFrame sein!")

    def forward(self, x):
        """
        Forward-Pass mit physikalischer Gleichung.
        Parameters
        ----------
        x : torch.Tensor oder pd.DataFrame
            Input-Daten der Form [seq_len, input_size] oder [batch, seq_len, input_size].
        """

        # Füge Batch-Dimension hinzu, falls nicht vorhanden
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [seq_len, 1, input_size]

        x = x.to(self.device)
        if not x.is_contiguous():
            x = x.contiguous()

        batch_size = x.size(1)  # Korrekte Batch-Größe (1 oder größer)
        seq_len = x.size(0)

        # Extrahiere a, v, F
        a, v, F = self._extract_physics_terms(x)

        # RNN-Pass mit korrekter Hidden-State-Initialisierung
        h0 = torch.zeros(
            self.n_hidden_layers,
            batch_size,  # Wichtig: Verwende die tatsächliche Batch-Größe!
            self.n_hidden_size,
            device=self.device
        )
        x_pinn = x[:, :, self.idx_v]
        if x_pinn.dim() == 2:
            x_pinn = x_pinn.unsqueeze(1)
        out, h_n = self.rnn(x, h0)  # out: [batch, seq_len, n_hidden_size]

        # Berechne z für jeden Zeitschritt
        z = self.z_layer(out)  # [batch, seq_len, z_dim]

        # Berechne dz/dt
        dz_dt = torch.zeros_like(z)
        dz_dt[:, 1:, :] = (z[:, 1:, :] - z[:, :-1, :]) / self.dt

        # Physikalische Gleichung (wie zuvor)
        a_expanded = a.expand(seq_len, -1, -1)
        v_expanded = v.expand(seq_len, -1, -1)
        F_expanded = F.expand(seq_len, -1, -1)

        y_physics = (
                self.c1 * a_expanded +
                self.c2 * v_expanded +
                self.c3 * F_expanded +
                self.c4 * z +
                self.c5 * dz_dt
        )
        out = self.fc(out)
        y_pred = y_physics + out
        return y_pred

    def train_model(self, X_train, y_train, X_val, y_val, target='curr_x', **kwargs):
        """
        Überschreibt train_model, um die Indizes von a, v, F zu bestimmen.
        Ruft dann die ursprüngliche train_model-Methode von BaseTorchModel auf.
        """
        # Bestimme die Indizes von a, v, F (nur beim ersten Aufruf)
        if self.idx_v is None or self.idx_a is None or self.idx_F is None:
            self._set_physics_indices(X_train, target)

        # Konvertiere X_train, y_train zu Tensoren (falls nötig)
        X_train_tensor = self.scaled_to_tensor(X_train)
        y_train_tensor = self.to_tensor(y_train)
        X_val_tensor = self.scaled_to_tensor(X_val)
        y_val_tensor = self.to_tensor(y_val)

        # Rufe die ursprüngliche train_model-Methode auf
        return super().train_model(
            X_train_tensor, y_train_tensor,
            X_val_tensor, y_val_tensor,
            **kwargs
        )

    def test_model(self, X, y_target, target='curr_x', **kwargs):
        """
        Überschreibt test_model, um die Indizes von a, v, F zu bestimmen.
        Ruft dann die ursprüngliche test_model-Methode von BaseTorchModel auf.
        """
        # Bestimme die Indizes von a, v, F (falls noch nicht geschehen)
        if self.idx_v is None or self.idx_a is None or self.idx_F is None:
            self._set_physics_indices(X, target)

        # Rufe die ursprüngliche test_model-Methode auf
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
        if 'physics_weight' in kwargs:
            self.physics_weight = kwargs['physics_weight']
        if 'axis' in kwargs:
            self.axis = kwargs['axis']

        if self.input_size is not None and any(param in kwargs for param in [
            'n_hidden_size', 'n_hidden_layers', 'activation', 'dropout_rate', 'z_dim'
        ]):
            self._initialize()

    def get_documentation(self):
        """Gibt eine Dokumentation des Modells zurück."""
        doc = {
            "model_type": "PhysicsInformedRNN",
            "hyperparameters": {
                "input_size": self.input_size,
                "output_size": self.output_size,
                "n_hidden_size": self.n_hidden_size,
                "n_hidden_layers": self.n_hidden_layers,
                "activation": self.activation_name,
                "learning_rate": self.learning_rate,
                "dropout_rate": self.dropout_rate,
                "z_dim": self.z_dim,
                "dt": self.dt,
                "physics_weight": self.physics_weight,
                "axis": self.axis,
                "physics_indices": {
                    "col_v": self.col_v if hasattr(self, 'col_v') else None,
                    "col_a": self.col_a if hasattr(self, 'col_a') else None,
                    "col_F": self.col_F if hasattr(self, 'col_F') else None,
                    "idx_v": self.idx_v,
                    "idx_a": self.idx_a,
                    "idx_F": self.idx_F,
                }
            },
            "learned_coefficients": {
                "c1": self.c1.item(),
                "c2": self.c2.item(),
                "c3": self.c3.item(),
                "c4": self.c4.item(),
                "c5": self.c5.item(),
            },
            "device": str(self.device),
        }
        return doc