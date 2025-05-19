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
import torch.jit
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer

import Models.model_base as mb


# Defines a configurable neural network
class Net(mb.BaseNetModel):
    def __init__(self, input_size, output_size, n_hidden_size, n_hidden_layers, activation=nn.ReLU, learning_rate=0.0001, name = "Neural_Net"):
        """
        Initializes a configurable neural network.

        Parameters
        ----------
        input_size : int
            The number of input features.
        output_size : int
            The number of output features.
        hidden_size : int
            The number of features in each hidden layer.
        n_hidden_layers : int
            The number of hidden layers in the network.
        activation : torch.nn.Module, optional
            The activation function to be used in the hidden layers. The default is nn.ReLU.

        Returns
        -------
        None
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, n_hidden_size)
        self.activation = activation()  # instantiate the activation function
        self.fcs = nn.ModuleList([nn.Linear(n_hidden_size, n_hidden_size) for _ in range(n_hidden_layers)])
        self.fc3 = nn.Linear(n_hidden_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        # Save parameter for documentation
        self.learning_rate = learning_rate
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.name = name

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
        x = self.activation(self.fc1(x))  # apply the activation function
        for fc in self.fcs:
            x = self.activation(fc(x))  # apply the activation function
        x = self.fc3(x)
        return x

def get_reference_net(input_size):
    """
    Get the reference neural network

    Parameters
    ----------
    input_size : int
        The number of input features.

    Returns
    -------
    Net(input_size, 1, input_size, n_hidden_layers=1, nn.ReLU, 0.001)
    """
    return Net(input_size, 1, input_size, 1, learning_rate=0.001)

class RNN(mb.BaseNetModel):
    def __init__(self, input_size, output_size, n_hidden_size, n_hidden_layers, activation=nn.ReLU, learning_rate=0.0001, name = "Recurrent_Neural_Net", batched_input=False):
        """
        Initializes a configurable recurrent neural network.

        Parameters
        ----------
        input_size : int
            The number of input features.
        output_size : int
            The number of output features.
        n_hidden_size : int
            The number of features in each hidden layer.
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
        self.rnn = nn.RNN(input_size, n_hidden_size, n_hidden_layers, batch_first=batched_input)
        self.fc = nn.Linear(n_hidden_size, output_size)
        self.activation = activation()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        # Save parameter for documentation
        self.learning_rate = learning_rate
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.name = name

    def forward(self, x):
        """
        Defines the forward pass of the recurrent neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the neural network.

        Returns
        -------
        x : torch.Tensor
            The output tensor from the neural network.
        """
        if x.dim() == 2:
            # Non batched input
            h0 = torch.zeros(self.n_hidden_layers, self.n_hidden_size).to(self.device)

        elif x.dim() == 3:
            # batched input
            h0 = torch.zeros(self.n_hidden_layers, x.size(0), self.n_hidden_size).to(self.device)

        # Forward pass through RNN layer
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out)

        return out

class LSTM(mb.BaseNetModel):
    def __init__(self, input_size, output_size, n_hidden_size, n_hidden_layers, activation=nn.ReLU, learning_rate=0.0001, name = "LSTM", batched_input=False):
        """
        Initializes a configurable recurrent neural network.

        Parameters
        ----------
        input_size : int
            The number of input features.
        output_size : int
            The number of output features.
        n_hidden_size : int
            The number of features in each hidden layer.
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
        self.rnn = nn.LSTM(input_size, n_hidden_size, n_hidden_layers, batch_first=batched_input)
        self.fc = nn.Linear(n_hidden_size, output_size)
        self.activation = activation()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        # Save parameter for documentation
        self.learning_rate = learning_rate
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.name = name

    def forward(self, x):
        """
        Defines the forward pass of the recurrent neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the neural network.

        Returns
        -------
        x : torch.Tensor
            The output tensor from the neural network.
        """
        if x.dim() == 2:
            # Non batched input
            h0 = torch.zeros(self.n_hidden_layers, self.n_hidden_size).to(self.device)

        elif x.dim() == 3:
            # batched input
            h0 = torch.zeros(self.n_hidden_layers, x.size(0), self.n_hidden_size).to(self.device)

        # Forward pass through RNN layer
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out)

        return out

class GRU(mb.BaseNetModel):
    def __init__(self, input_size, output_size, n_hidden_size, n_hidden_layers, activation=nn.ReLU, learning_rate=0.0001, name = "GRU", batched_input=False):
        """
        Initializes a configurable recurrent neural network.

        Parameters
        ----------
        input_size : int
            The number of input features.
        output_size : int
            The number of output features.
        n_hidden_size : int
            The number of features in each hidden layer.
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
        self.rnn = nn.GRU(input_size, n_hidden_size, n_hidden_layers, batch_first=batched_input)
        self.fc = nn.Linear(n_hidden_size, output_size)
        self.activation = activation()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        # Save parameter for documentation
        self.learning_rate = learning_rate
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.name = name

    def forward(self, x):
        """
        Defines the forward pass of the recurrent neural network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the neural network.

        Returns
        -------
        x : torch.Tensor
            The output tensor from the neural network.
        """
        if x.dim() == 2:
            # Non batched input
            h0 = torch.zeros(self.n_hidden_layers, self.n_hidden_size).to(self.device)

        elif x.dim() == 3:
            # batched input
            h0 = torch.zeros(self.n_hidden_layers, x.size(0), self.n_hidden_size).to(self.device)

        # Forward pass through RNN layer
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out)

        return out

# Defines a configurable neural network
class SequentialNet(mb.BaseNetModel):
    def __init__(self, input_size, output_size, n_hidden_size, n_hidden_layers, n_rnn_layer = 1, activation=nn.PReLU, learning_rate=0.01, name = "Sequential_Neural_Net"):
        """
        Initializes a configurable neural network.

        Parameters
        ----------
        input_size : int
            The number of input features.
        output_size : int
            The number of output features.
        hidden_size : int
            The number of features in each hidden layer.
        n_hidden_layers : int
            The number of hidden layers in the network.
        activation : torch.nn.Module, optional
            The activation function to be used in the hidden layers. The default is nn.ReLU.

        Returns
        -------
        None
        """
        super(SequentialNet, self).__init__()
        self.fc_in = nn.Linear(input_size, n_hidden_size)
        self.activation = activation()  # instantiate the activation function
        self.fcs = nn.ModuleList([nn.Linear(n_hidden_size, n_hidden_size) for _ in range(n_hidden_layers)])
        self.n_rnn_size = 2*output_size
        self.fc_connection = nn.Linear(n_hidden_size,  self.n_rnn_size)
        self.rnn = nn.RNN(self.n_rnn_size, self.n_rnn_size, n_rnn_layer) # TODO: Parametrisierbar gestalten
        self.fc_end_connection = nn.Linear(self.n_rnn_size, self.n_rnn_size)
        self.fc_out = nn.Linear(self.n_rnn_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        # Save parameter for documentation
        self.learning_rate = learning_rate
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.n_rnn_layer = n_rnn_layer
        self.name = name

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
class GruNet(mb.BaseNetModel):
    def __init__(self, input_size, output_size, n_hidden_size, n_hidden_layers, n_rnn_layer = 1, activation=nn.PReLU, learning_rate=0.01, name = "GruNet"):
        """
        Initializes a configurable neural network.

        Parameters
        ----------
        input_size : int
            The number of input features.
        output_size : int
            The number of output features.
        hidden_size : int
            The number of features in each hidden layer.
        n_hidden_layers : int
            The number of hidden layers in the network.
        activation : torch.nn.Module, optional
            The activation function to be used in the hidden layers. The default is nn.ReLU.

        Returns
        -------
        None
        """
        super(SequentialNet, self).__init__()
        self.fc_in = nn.Linear(input_size, n_hidden_size)
        self.activation = activation()  # instantiate the activation function
        self.fcs = nn.ModuleList([nn.Linear(n_hidden_size, n_hidden_size) for _ in range(n_hidden_layers)])
        self.n_rnn_size = 2*output_size
        self.fc_connection = nn.Linear(n_hidden_size,  self.n_rnn_size)
        self.rnn = nn.GRU(self.n_rnn_size, self.n_rnn_size, n_rnn_layer) # TODO: Parametrisierbar gestalten
        self.fc_end_connection = nn.Linear(self.n_rnn_size, self.n_rnn_size)
        self.fc_out = nn.Linear(self.n_rnn_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        # Save parameter for documentation
        self.learning_rate = learning_rate
        self.n_hidden_size = n_hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.n_rnn_layer = n_rnn_layer
        self.name = name

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
class QuantileIdNetModel(Net):
    def __init__(self, input_size, output_size, n_neurons, n_layers, activation=nn.ReLU, output_distribution='uniform'):
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
        super(QuantileIdNetModel, self).__init__(input_size * 2, output_size, n_neurons, n_layers, activation)
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
                 max_bins=64):
        super().__init__(input_size, max_bins, n_neurons, n_layers, activation)
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.bins = None
        self.loss_fn = nn.CrossEntropyLoss()

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
                f'Epoch {epoch + 1}/{n_epochs}, Val Error: {val_error:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.load_state_dict(best_model_state)
        return best_val_error

    def test_model(self, X, y_target):
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
        loss = self.loss_fn(torch.tensor(pred_probs), torch.tensor(y_target_discrete))

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

    def test_model(self, X, y_target):
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
        loss = self.loss_fn(torch.tensor(pred_probs), torch.tensor(y_target_discrete))

        return loss.item(), y_pred
