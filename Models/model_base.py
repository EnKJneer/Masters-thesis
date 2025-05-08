import numpy as np
import optuna
import torch
import torch.jit
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def criterion(self,y_target, y_pred):
        """
        abstractmethod
        Compute the loss between the target and predicted values.

        Parameters:
        y_target (Tensor): The target values.
        y_pred (Tensor): The predicted values.

        Returns:
        Tensor: The computed loss.
        """
        pass
    @abstractmethod
    def predict(self,X):
        """
        abstractmethod
        Make predictions based on the input data.

        Parameters:
        X (Tensor): The input data.

        Returns:
        Tensor: The predicted values.
        """
        pass
    @abstractmethod
    def train_model(self, X_train, y_train, X_val, y_val):       
        """
        abstractmethod
        Train the model using the training data and validate it using the validation data.

        Parameters:
        X_train (Tensor): The training input data.
        y_train (Tensor): The training target values.
        X_val (Tensor): The validation input data.
        y_val (Tensor): The validation target values.
        """
        pass
    @abstractmethod
    def test_model(self,X,y_target):
        """
        abstractmethod
        Test the model using the test data and compute the loss.

        Parameters:
        X (Tensor): The test input data.
        y_target (Tensor): The test target values.

        Returns:
        tuple: A tuple containing the loss and the predicted values.
        """
        pass

class BaseNetModel(BaseModel, nn.Module):
    @abstractmethod
    def forward(self, x):
        """
        abstractmethod
        Define the forward pass of the network.
        This must be implemented by subclasses.

        Parameters:
        x (Tensor): The input data.

        Returns:
        Tensor: The output of the forward pass.
        """
        pass
    
    def criterion(self, y_target, y_pred):
        """
        Compute the Mean Squared Error (MSE) loss between the target and predicted values.

        Parameters:
        y_target (Tensor): The target values.
        y_pred (Tensor): The predicted values.

        Returns:
        Tensor: The computed MSE loss.
        """
        criterion = nn.MSELoss()
        return criterion(y_target.squeeze(), y_pred.squeeze())
    
    def predict(self,X):
        """
        Make predictions based on the input data using the forward pass.

        Parameters:
        X (Tensor): The input data.

        Returns:
        Tensor: The predicted values.
        """
        return self(X)
    
    def test_model(self,X,y_target):
        """
        Test the model using the test data and compute the loss.

        Parameters:
        X (Tensor): The test input data.
        y_target (Tensor): The test target values.

        Returns:
        tuple: A tuple containing the loss and the predicted values.
        """
        assert(self.scaler is None, "model musst first be trained")
        # Scaling the input data
        X_scaled = self.scaler.transform(X)
        
        # Convert input data to PyTorch tensors
        X = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        if not isinstance(y_target, np.ndarray):
            y_target = y_target.to_numpy()
        y_target = torch.tensor(y_target, dtype=torch.float32).to(self.device)
        y_pred = self.predict(X)
        loss = self.criterion(y_target, y_pred)

        return loss.item(), y_pred.detach().cpu().numpy()

    def train_model(self, X_train, y_train, X_val, y_val, learning_rate, n_epochs=100, patience=20, draw_loss=False, epsilon = 0.0001, trial=None):
        """
        Trains a neural network using the provided training and validation data with early stopping, dynamic learning rate adjustment, and Optuna pruning.

        Parameters
        ----------
        X_train : array-like
            The training input data.
        y_train : array-like
            The training target data.
        X_val : array-like
            The validation input data.
        y_val : array-like
            The validation target data.
        learning_rate : float
            The initial learning rate for the optimizer.
        n_epochs : int, optional
            The maximum number of training epochs. The default is 100.
        patience : int, optional
            The number of epochs to wait for improvement in validation loss before stopping early. Default is 20.
        draw_loss : bool, optional
            If True, plots training and validation loss after each epoch. Default is False.
        epsilon : float, optional
            The minimum improvement in validation loss required to reset the patience counter. Default is 0.0001.
        trial : optuna.trial.Trial, optional
            An Optuna trial object used to suggest values for the hyperparameters and perform pruning. If provided, the function will use the trial object for pruning based on intermediate validation errors.

        Returns
        -------
        best_val_error : float
            The best validation error achieved during training.
        model : nn.Module
            The trained neural network model with the best weights loaded.
        """
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        # Scaling the input data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scaler = scaler

        # Convert input data to PyTorch tensors
        X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        if not isinstance(y_train, np.ndarray):
            y_train = y_train.to_numpy()
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        if not isinstance(y_val, np.ndarray):
            y_val = y_val.to_numpy()
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        # Define the optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Use a learning rate scheduler to reduce the learning rate when validation loss plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        # Initialize lists for tracking losses if plotting is enabled
        if draw_loss:
            loss_vals, epochs, loss_train = [], [], []
            fig, ax = plt.subplots()
            line_val, = ax.plot(epochs, loss_vals, 'r-', label='validation')
            line_train, = ax.plot(epochs, loss_train, 'b-', label='training')
            ax.legend()
            
        best_val_error = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(n_epochs):
            self.train()
            optimizer.zero_grad()
            
            # Forward pass and loss computation on training data
            y_pred = self(X_train)
            loss = self.criterion(y_pred, y_train)
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Evaluate on validation data
            self.eval()
            with torch.no_grad():
                y_val_pred = self(X_val)
                val_error = self.criterion(y_val_pred, y_val).item()
                
                # Check if the current validation error is the best so far
                if val_error < best_val_error - epsilon:
                    best_val_error = val_error
                    best_model_state = self.state_dict()  # Save the best model weights
                    patience_counter = 0  # Reset patience counter
                elif epoch > (n_epochs / 10) and epoch > 10:
                    patience_counter += 1

                # Call the scheduler to adjust the learning rate if needed
                scheduler.step(val_error)

                # Check for early stopping condition
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                # Pruning logic
                if trial is not None:                
                    trial.report(val_error, step=epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                    
                # Plot losses if enabled
                if draw_loss:
                    epochs.append(epoch)
                    loss_vals.append(val_error)
                    loss_train.append(loss.item())
                    line_val.set_xdata(epochs)
                    line_val.set_ydata(loss_vals)
                    line_train.set_xdata(epochs)
                    line_train.set_ydata(loss_train)
                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.001)

            print(f'Epoch {epoch+1}/{n_epochs}, Val Error: {val_error:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Finalize plotting
        if draw_loss:
            plt.ioff()
            plt.show()

        # Load the model with the best validation performance
        self.load_state_dict(best_model_state)
        return best_val_error