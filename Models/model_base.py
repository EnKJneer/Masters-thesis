import numpy as np
import optuna
import pandas as pd
import torch
import torch.jit
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def criterion(self, y_target, y_pred):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def train_model(self, X_train, y_train, X_val, y_val):
        pass

    @abstractmethod
    def test_model(self, X, y_target):
        pass

class BaseNetModel(BaseModel, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    def criterion(self, y_target, y_pred):
        criterion = nn.MSELoss()
        return criterion(y_target.squeeze(), y_pred.squeeze())

    def predict(self, X):
        return self(X)

    def scale_data(self, X):
        """
        Scale the input data using the scaler fitted during training.

        Parameters:
        X (Tensor): The input data.

        Returns:
        Tensor: The scaled input data.
        """
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        return self.scaler.transform(X)

    def test_model(self, X, y_target):
        X_scaled = self.scale_data(X)
        X = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        if not isinstance(y_target, np.ndarray):
            y_target = y_target.to_numpy()
        y_target = torch.tensor(y_target, dtype=torch.float32).to(self.device)
        y_pred = self.predict(X)
        loss = self.criterion(y_target, y_pred)
        return loss.item(), y_pred.detach().cpu().numpy()

    def train_model(self, X_train, y_train, X_val, y_val, learning_rate=0.0001, n_epochs=100, patience=20, draw_loss=False, epsilon=0.0001, trial=None, n_outlier=12):
        print(self.device)

        X_train_scaled = self.scale_data(X_train)
        X_val_scaled = self.scale_data(X_val)

        X_train = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        if not isinstance(y_train, np.ndarray):
            y_train = y_train.to_numpy()
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        if not isinstance(y_val, np.ndarray):
            y_val = y_val.to_numpy()
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        if draw_loss:
            loss_vals, epochs, loss_train = [], [], []
            fig, ax = plt.subplots()
            line_val, = ax.plot(epochs, loss_vals, 'r-', label='validation')
            line_train, = ax.plot(epochs, loss_train, 'b-', label='training')
            ax.legend()

        best_val_error = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(X_train)
            loss = self.criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                y_val_pred = self(X_val)
                val_error = self.criterion(y_val_pred, y_val).item()

                if val_error < best_val_error - epsilon:
                    best_val_error = val_error
                    best_model_state = self.state_dict()
                    patience_counter = 0
                elif epoch > (n_epochs / 10) and epoch > 10:
                    patience_counter += 1

                scheduler.step(val_error)

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

                if trial is not None:
                    trial.report(val_error, step=epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

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

        if draw_loss:
            plt.ioff()
            plt.show()

        self.load_state_dict(best_model_state)
        return best_val_error