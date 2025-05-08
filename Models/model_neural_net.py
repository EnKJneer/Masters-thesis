# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:20:38 2024

@author: Jonas Kyrion

SKRIPT DESCRIPTION:
    
Contains parameterizable neural network with necessary training functions
"""
import torch
import torch.jit
import torch.nn as nn

import Models.model_base as mb

# Defines a configurable neural network
class Net(mb.BaseNetModel):
    def __init__(self, input_size, output_size, n_neurons, n_layers, activation=nn.ReLU):
        """
        Initializes a configurable neural network.

        Parameters
        ----------
        input_size : int
            The number of input features.
        output_size : int
            The number of output features.
        n_neurons : int
            The number of neurons in each hidden layer.
        n_layers : int
            The number of hidden layers in the network.
        activation : torch.nn.Module, optional
            The activation function to be used in the hidden layers. The default is nn.ReLU.

        Returns
        -------
        None
        """      
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, n_neurons)
        self.activation = activation()  # instantiate the activation function
        self.fcs = nn.ModuleList([nn.Linear(n_neurons, n_neurons) for _ in range(n_layers - 2)])
        self.fc3 = nn.Linear(n_neurons, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        x = self.activation(self.fc1(x))  # apply the activation function
        for fc in self.fcs:
            x = self.activation(fc(x))  # apply the activation function
        x = self.fc3(x)
        return x
