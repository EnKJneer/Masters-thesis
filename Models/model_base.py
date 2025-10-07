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

import matplotlib
matplotlib.use('tkagg')

class BaseModel(ABC):
    @abstractmethod
    def __init__(self, name):
        self.name = name
    @abstractmethod
    def criterion(self, y_target, y_pred):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        pass

    @abstractmethod
    def test_model(self, X, y_target):
        pass
    @abstractmethod
    def get_documentation(self):
        pass
    @abstractmethod
    def reset_hyperparameter(self):
        pass

class BaseTorchModel(BaseModel, nn.Module):
    def __init__(self, input_size=None, output_size=1, name="BaseNetModel", learning_rate=0.001, optimizer_type='adam', dropout_rate=0.0):
        """
        Initializes the base neural network model with common attributes.

        Parameters
        ----------
        input_size : int, optional
            The number of input features. If None, it will be set during the first training call.
        output_size : int
            The number of output features.
        name : str
            The name of the model.
        learning_rate : float
            The learning rate for the optimizer.
        optimizer_type : str
            The type of optimizer to use.
        """
        BaseModel.__init__(self, name)  # Initialize BaseModel
        nn.Module.__init__(self)  # Initialize nn.Module
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None
        self.dropout_rate = dropout_rate

        # Map the activation function name to the corresponding PyTorch function
        self.activation_map = {
            'ReLU': nn.ReLU,
            'Sigmoid': nn.Sigmoid,
            'Tanh': nn.Tanh,
            'ELU': nn.ELU,
        }

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def criterion_Huber(self, y_target, y_pred, delta = 0.22):
        criterion = nn.HuberLoss(delta=delta)
        return criterion(y_target.squeeze(), y_pred.squeeze())

    def criterion(self, y_target, y_pred):
        criterion = nn.MSELoss()
        return criterion(y_target.squeeze(), y_pred.squeeze())

    def predict(self, X) -> np.ndarray:
        if type(X) is not torch.Tensor:
            X = self.scaled_to_tensor(X)
        return self(X).cpu().detach().numpy().squeeze()

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

    def test_model(self, x, y_target, criterion_test = None):
        if criterion_test is None:
            criterion_test = self.criterion
        X_scaled = self.scale_data(x)
        X = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        y_pred_num = self.predict(X)
        y_pred = torch.tensor(y_pred_num, dtype=torch.float32).to(self.device)
        if type(y_target) is not torch.Tensor:
            if type(y_target) is not np.array:
                y_target = np.array(y_target)
            y_target = torch.tensor(y_target, dtype=torch.float32).to(self.device)

        loss = criterion_test(y_target, y_pred)

        return loss.item(), y_pred_num

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

    def to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif hasattr(data, 'values'):
            return torch.tensor(data.values, dtype=torch.float32).to(self.device)
        else:
            # Falls numpy array oder anderes
            return torch.tensor(data, dtype=torch.float32).to(self.device)

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
                        loss = self.criterion(batch_y_tensor, output)
                        loss.backward()
                        loss_total += loss
                    return loss_total
                else:
                    x_tensor = self.scaled_to_tensor(X_train)
                    y_tensor = self.to_tensor(y_train)
                    output = self(x_tensor)
                    loss = self.criterion(y_tensor, output)
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
                        loss = self.criterion(output, batch_y_tensor)
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
                        val_loss = self.criterion(batch_y_tensor, output)
                        val_losses.append(val_loss.item())
                else:
                    x_tensor = self.scaled_to_tensor(X_val)
                    y_tensor = self.to_tensor(y_val)

                    output = self(x_tensor)
                    val_loss = self.criterion(y_tensor, output)
                    val_losses.append(val_loss.item())

            n = max(1, len(train_losses))
            avg_train_loss = sum(train_losses) / n
            n = max(1, len(val_losses))
            avg_val_loss = sum(val_losses) / n

            if avg_val_loss < best_val_error - epsilon:
                best_val_error = avg_val_loss
                best_model_state = self.state_dict()
                patience_counter = 0
            elif epoch > (n_epochs / 100):
                patience_counter += 1

            if np.isnan(avg_val_loss):
                # Neustart
                print(f"Validation loss: {avg_val_loss} --- Restart training")
                self._initialize()
                epoch = 0

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

    def get_documentation(self):
        documentation = {"hyperparameters": {
            "learning_rate": self.learning_rate,
            "n_hidden_size": self.n_hidden_size,
            "n_hidden_layers": self.n_hidden_layers,
            "activation": self.activation.__class__.__name__,
            "optimizer_type": self.optimizer_type,
            "dropout_rate": self.dropout_rate,
        }}
        return documentation

