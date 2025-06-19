import numpy as np
import optuna
import pandas as pd
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
import optax
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional, Callable
import pickle
import flax.linen as nn
import optax
from flax.training import train_state
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Any, Callable, Sequence
from abc import ABC
import Models.JAX_Version.model_base as jmb

class Net(jmb.BaseNetModel):
    def __init__(self, input_size=None, output_size=1, n_hidden_size=None, n_hidden_layers=1, activation="relu",
                 learning_rate=0.001, name="Neural_Net_JAX", optimizer_type='adam'):
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
        activation : str or callable, optional
            The activation function to be used in the hidden layers. The default is "relu".
            Can be string ("relu", "tanh", "sigmoid") or a JAX activation function.
        learning_rate : float
            The learning rate for the optimizer.
        name : str
            The name of the model.
        """
        super(Net, self).__init__(input_size=input_size, output_size=output_size, name=name,
                                  learning_rate=learning_rate, optimizer_type=optimizer_type)
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers

        # Handle activation function - support both strings and PyTorch-like objects
        if hasattr(activation, '__class__') and hasattr(activation.__class__, '__name__'):
            # PyTorch activation object passed
            activation_name = activation.__class__.__name__.lower()
            if 'relu' in activation_name:
                self.activation = jax.nn.relu
                self.activation_name = "ReLU"
            elif 'tanh' in activation_name:
                self.activation = jax.nn.tanh
                self.activation_name = "Tanh"
            elif 'sigmoid' in activation_name:
                self.activation = jax.nn.sigmoid
                self.activation_name = "Sigmoid"
            else:
                self.activation = jax.nn.relu
                self.activation_name = "ReLU"
        elif isinstance(activation, str):
            # String activation
            if activation.lower() == "relu":
                self.activation = jax.nn.relu
                self.activation_name = "ReLU"
            elif activation.lower() == "tanh":
                self.activation = jax.nn.tanh
                self.activation_name = "Tanh"
            elif activation.lower() == "sigmoid":
                self.activation = jax.nn.sigmoid
                self.activation_name = "Sigmoid"
            else:
                self.activation = jax.nn.relu
                self.activation_name = "ReLU"
        else:
            # Assume it's a callable activation function
            self.activation = activation
            self.activation_name = str(activation)

        # Initialize layers only if input_size is provided
        if self.input_size is not None:
            self._initialize()

    def _initialize(self):
        """Initialize the parameters of the neural network."""
        self.scaler = None
        if self.n_hidden_size is None:
            self.n_hidden_size = self.input_size

        # Initialize parameters with Xavier/Glorot initialization (matching PyTorch default)
        self.key, *subkeys = random.split(self.key, num=2 + self.n_hidden_layers + 1)

        # Input layer - Xavier initialization
        fan_in, fan_out = self.input_size, self.n_hidden_size
        bound = jnp.sqrt(6.0 / (fan_in + fan_out))
        w1 = random.uniform(subkeys[0], (self.input_size, self.n_hidden_size), minval=-bound, maxval=bound)
        b1 = jnp.zeros((self.n_hidden_size,))

        # Hidden layers
        hidden_weights = []
        hidden_biases = []
        for i in range(self.n_hidden_layers):
            fan_in, fan_out = self.n_hidden_size, self.n_hidden_size
            bound = jnp.sqrt(6.0 / (fan_in + fan_out))
            w = random.uniform(subkeys[1 + i], (self.n_hidden_size, self.n_hidden_size), minval=-bound, maxval=bound)
            b = jnp.zeros((self.n_hidden_size,))
            hidden_weights.append(w)
            hidden_biases.append(b)

        # Output layer
        fan_in, fan_out = self.n_hidden_size, self.output_size
        bound = jnp.sqrt(6.0 / (fan_in + fan_out))
        w_out = random.uniform(subkeys[-1], (self.n_hidden_size, self.output_size), minval=-bound, maxval=bound)
        b_out = jnp.zeros((self.output_size,))

        self.params = {
            'w1': w1,
            'b1': b1,
            'hidden_weights': hidden_weights,
            'hidden_biases': hidden_biases,
            'w_out': w_out,
            'b_out': b_out
        }

    def forward(self, params, x):
        """
        Defines the forward pass of the neural network.

        Parameters
        ----------
        params : dict
            The parameters of the neural network.
        x : jnp.ndarray
            The input tensor to the neural network.

        Returns
        -------
        x : jnp.ndarray
            The output tensor from the neural network.
        """
        if self.input_size is None:
            self.input_size = x.shape[1]
            if self.n_hidden_size is None:
                self.n_hidden_size = self.input_size
            self._initialize()

        # Input layer
        x = self.activation(jnp.dot(x, params['w1']) + params['b1'])

        # Hidden layers
        for w, b in zip(params['hidden_weights'], params['hidden_biases']):
            x = self.activation(jnp.dot(x, w) + b)

        # Output layer
        x = jnp.dot(x, params['w_out']) + params['b_out']

        return x