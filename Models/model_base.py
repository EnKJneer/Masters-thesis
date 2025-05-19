import numpy as np
import optuna
import pandas as pd
import torch
import torch.jit
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
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
        if type(X) is pd.DataFrame:
            X = X.values
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

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=100, patience=20, draw_loss=False, epsilon=0.0001,
                    trial=None, n_outlier=12):
        def scaled_to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            elif hasattr(data, 'values'):
                data_scaled = self.scale_data(data.values)
                return torch.tensor(data_scaled, dtype=torch.float32).to(self.device)
            else:
                # Falls numpy array oder anderes
                data_scaled = self.scale_data(data)
                return torch.tensor(data_scaled, dtype=torch.float32).to(self.device)
          
        def to_tensor(data):
                if isinstance(data, torch.Tensor):
                    return data.to(self.device)
                elif hasattr(data, 'values'):
                    return torch.tensor(data.values, dtype=torch.float32).to(self.device)
                else:
                    # Falls numpy array oder anderes
                    return torch.tensor(data, dtype=torch.float32).to(self.device)

        # Prüfen, ob Inputs Listen sind
        is_batched_train = isinstance(X_train, list) and isinstance(y_train, list)
        is_batched_val = isinstance(X_val, list) and isinstance(y_val, list)

        assert (not is_batched_train) or (len(X_train) == len(y_train)), "Trainingslist must have the same length"
        assert (not is_batched_val) or (len(X_val) == len(y_val)), "Validierungslisten must have the same length"

        print(f"Device: {self.device} | Batched: {is_batched_train}")

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
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

            train_losses = []

            if is_batched_train:
                for batch_x, batch_y in zip(X_train, y_train):
                    optimizer.zero_grad()
                    batch_x_tensor = scaled_to_tensor(batch_x)
                    batch_y_tensor = to_tensor(batch_y)

                    output = self(batch_x_tensor)
                    loss = self.criterion(output, batch_y_tensor)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
            else:
                optimizer.zero_grad()
                x_tensor = scaled_to_tensor(X_train)
                y_tensor = to_tensor(y_train)

                output = self(x_tensor)
                loss = self.criterion(output, y_tensor)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            self.eval()
            val_losses = []
            with torch.no_grad():
                if is_batched_val:
                    for batch_x, batch_y in zip(X_val, y_val):
                        batch_x_tensor = scaled_to_tensor(batch_x)
                        batch_y_tensor = to_tensor(batch_y)

                        output = self(batch_x_tensor)
                        val_loss = self.criterion(output, batch_y_tensor)
                        val_losses.append(val_loss.item())
                else:
                    x_tensor = scaled_to_tensor(X_val)
                    y_tensor = to_tensor(y_val)

                    output = self(x_tensor)
                    val_loss = self.criterion(output, y_tensor)
                    val_losses.append(val_loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)

            if avg_val_loss < best_val_error - epsilon:
                best_val_error = avg_val_loss
                best_model_state = self.state_dict()
                patience_counter = 0
            elif epoch > (n_epochs / 10) and epoch > 10:
                patience_counter += 1

            scheduler.step(avg_val_loss)

            if patience_counter >= patience:
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
                line_val.set_xdata(epochs)
                line_val.set_ydata(loss_vals)
                line_train.set_xdata(epochs)
                line_train.set_ydata(loss_train)
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)

            print(f'{self.name}: Epoch {epoch + 1}/{n_epochs}, Train Loss; {avg_train_loss:.4f} Val Error: {avg_val_loss:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        if draw_loss:
            plt.ioff()
            plt.show()

        self.load_state_dict(best_model_state)
        return best_val_error
    """

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=100, patience=20,
                    draw_loss=False, epsilon=0.0001, trial=None, n_outlier=12):
        print(self.device)

        def scale_padded_list(X_list):
            # Alle Einträge (DataFrames) zu einem großen DataFrame stapeln
            X_all = pd.concat(X_list, ignore_index=True)
            X_scaled = self.scale_data(X_all)  # Skaliert per StandardScaler
            # Jetzt wieder zurück in Einzelteile aufteilen
            split_sizes = [len(X) for X in X_list]
            arrays = np.split(X_scaled, np.cumsum(split_sizes)[:-1])
            return arrays

        def prepare_padded_batch(X_list, y_list):
            X_tensors = [torch.tensor(self.scale_data(X) , dtype=torch.float32) for X in X_list]
            y_tensors = [torch.tensor(y.to_numpy() if not isinstance(y, np.ndarray) else y, dtype=torch.float32) for y
                         in y_list]

            lengths = torch.tensor([x.shape[0] for x in X_tensors])
            X_padded = pad_sequence(X_tensors, batch_first=True)
            y_padded = pad_sequence(y_tensors, batch_first=True)

            mask = torch.arange(X_padded.size(1))[None, :] < lengths[:, None]
            return X_padded.to(self.device), y_padded.to(self.device), mask.to(self.device)

        def prepare_batch(X, y):
            X_scaled = self.scale_data(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            y_tensor = torch.tensor(y.to_numpy() if not isinstance(y, np.ndarray) else y, dtype=torch.float32).to(
                self.device)
            return X_tensor, y_tensor

        is_batched_train = isinstance(X_train, list)
        is_batched_val = isinstance(X_val, list)
        print(f"Training data batched: {is_batched_train}")
        print(f"Validation data batched: {is_batched_val}")
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
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

            if is_batched_train:
                X_tensor, y_tensor, mask = prepare_padded_batch(X_train, y_train)
                y_pred = self(X_tensor)
                loss_matrix = self.criterion(y_pred, y_tensor)
                if loss_matrix.dim() > 1:
                    loss_matrix = loss_matrix.mean(dim=-1)
                loss = (loss_matrix * mask).sum() / mask.sum()
                loss.backward()
                optimizer.step()
                avg_train_loss = loss.item()
            else:
                X_tensor, y_tensor = prepare_batch(X_train, y_train)
                y_pred = self(X_tensor)
                loss = self.criterion(y_pred, y_tensor)
                loss.backward()
                optimizer.step()
                avg_train_loss = loss.item()

            self.eval()
            with torch.no_grad():
                if is_batched_val:
                    X_tensor, y_tensor, mask = prepare_padded_batch(X_val, y_val)
                    y_val_pred = self(X_tensor)
                    val_loss_matrix = self.criterion(y_val_pred, y_tensor)
                    if val_loss_matrix.dim() > 1:
                        val_loss_matrix = val_loss_matrix.mean(dim=-1)
                    val_error = (val_loss_matrix * mask).sum().item() / mask.sum().item()
                else:
                    X_tensor, y_tensor = prepare_batch(X_val, y_val)
                    y_val_pred = self(X_tensor)
                    val_error = self.criterion(y_val_pred, y_tensor).item()

                if val_error < best_val_error - epsilon:
                    best_val_error = val_error
                    best_model_state = self.state_dict()
                    patience_counter = 0
                elif epoch > (n_epochs / 10) and epoch > 10:
                    patience_counter += 1

                scheduler.step(val_error)

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

                if trial is not None:
                    trial.report(val_error, step=epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                if draw_loss:
                    epochs.append(epoch)
                    loss_vals.append(val_error)
                    loss_train.append(avg_train_loss)
                    line_val.set_xdata(epochs)
                    line_val.set_ydata(loss_vals)
                    line_train.set_xdata(epochs)
                    line_train.set_ydata(loss_train)
                    ax.relim()
                    ax.autoscale_view()
                    plt.draw()
                    plt.pause(0.001)

            print(
                f'Epoch {epoch + 1}/{n_epochs}, Val Error: {val_error:.4f}, Train Loss: {avg_train_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if draw_loss:
            plt.ioff()
            plt.show()

        self.load_state_dict(best_model_state)
        return best_val_error


    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=100, patience=20, draw_loss=False, epsilon=0.0001, trial=None, n_outlier=12):
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

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
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
            loss = self.criterion(y_train, y_pred)
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
        return best_val_error"""