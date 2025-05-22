import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import optuna
import Models.model_base as mb

material_parameters = {
    "AL-2007-T4": {
        "hardness": 320, #Quelle; https://datenblaetter.thyssenkrupp.ch/en_aw_2007_0618.pdf Umrechung; https://www.bossard.com/de-de/assembly-technology-expert/technische-informationen-und-tools/online-konverter-und-rechner/harteumrechner/
        "tensile_strength": 370,
        "yield_strength": 250,
        "thermal_conductivity": 145,
        "modulus_of_elasticity": 72.5,
        "density": 2.85
    },
    "S235JRC": {
        "hardness": 120,
        "tensile_strength": 570,
        "yield_strength": 300,
        "thermal_conductivity": 54,
        "modulus_of_elasticity": 212,
        "density": 7.85
    }
}

class ModelZhou(mb.BaseModel, nn.Module):
    def __init__(self, learning_rate=0.0001, name="Phys_Model_Zhou", optimizer=optim.LBFGS):
        """
        Initializes the model from Zhou with the given parameters.
        Source: An improved cutting power model of machine tools in milling process Zhou et. al. 2016

        Parameters
        ----------
        learning_rate : float
            The learning rate for the optimizer.
        name : str
            The name of the model.
        optimizer : torch.optim.Optimizer, optional
            The optimizer to use for training. Default is Adam.
        """
        super(ModelZhou, self).__init__()
        self.learning_rate = learning_rate
        self.name = name
        self.optimizer_class = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.scaler = None

        # Initialize the parameters as a matrix
        self.theta = nn.Parameter(torch.randn(3))

    def forward(self, x, H):
        """
        Defines the forward pass of the custom model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the model.

        Returns
        -------
        x : torch.Tensor
            The output tensor from the model.
        """
        MRR = x[:, 0]
        n = x[:, 1]
        v_x = x[:, 2]
        v_y = x[:, 3]
        output_mat = 0.0012 * (n**-0.0360) * (H**0.1773) * MRR
        output_spin = self.theta[0] * n
        output_feed = self.theta[1] * torch.sqrt(v_x**2 + v_y**2)
        output = output_mat + output_spin + output_feed + self.theta[2]
        return output

    def criterion(self, y_target, y_pred):
        criterion = nn.MSELoss()
        return criterion(y_target.squeeze(), y_pred.squeeze())

    def predict(self, X):
        return self(X)

    def test_model(self, X, y_target):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if not isinstance(y_target, np.ndarray):
            y_target = y_target.to_numpy()
        y_target = torch.tensor(y_target, dtype=torch.float32).to(self.device)
        y_pred = self.predict(X)
        loss = self.criterion(y_target, y_pred)
        return loss.item(), y_pred.detach().cpu().numpy()

    def train_model(self, X_train_tuple, y_train, X_val_tuple, y_val, n_epochs=100, patience=20, draw_loss=False,
                    epsilon=0.0001,
                    trial=None, n_outlier=12):
        """
        Train the model.

        Parameters
        ----------
        X_train_tuple : tuple
            The training input samples and additional input H.
        y_train : array-like
            The target values.
        X_val_tuple : tuple
            The validation input samples and additional input H.
        y_val : array-like
            The validation target values.
        n_epochs : int
            The number of epochs to train the model.
        patience : int
            The number of epochs to wait before early stopping if no improvement.
        draw_loss : bool
            Whether to draw the loss during training.
        epsilon : float
            The minimum improvement required to consider as an improvement.
        trial : optuna.Trial, optional
            The Optuna trial object.
        n_outlier : int
            The number of outliers to consider.

        Returns
        -------
        best_val_error : float
            The best validation error achieved during training.
        """
        X_train = X_train_tuple[0]
        H_train = X_train_tuple[1]
        X_val = X_val_tuple[0]
        H_val = X_val_tuple[1]

        def get_x(df):
            prefix_current = '_1_current'
            return df[['materialremoved_sim' + prefix_current, 'v_sp' + prefix_current, 'v_x' + prefix_current,
                       'v_y' + prefix_current]]

        def to_tensor(data):
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            elif hasattr(data, 'values'):
                return torch.tensor(data.values, dtype=torch.float32).to(self.device)
            else:
                return torch.tensor(data, dtype=torch.float32).to(self.device)

        is_batched_train = isinstance(X_train, list) and isinstance(y_train, list)
        is_batched_val = isinstance(X_val, list) and isinstance(y_val, list)

        assert (not is_batched_train) or (len(X_train) == len(y_train)), "Trainingslist must have the same length"
        assert (not is_batched_val) or (len(X_val) == len(y_val)), "Validierungslist must have the same length"

        print(f"Device: {self.device} | Batched: {is_batched_train}")

        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        if draw_loss:
            loss_vals, epochs, loss_train = [], [], []
            fig, ax = plt.subplots()
            line_val, = ax.plot(epochs, loss_vals, 'r-', label='validation')
            line_train, = ax.plot(epochs, loss_train, 'b-', label='training')
            ax.legend()

        best_val_error = float('inf')
        patience_counter = 0

        def closure():
            optimizer.zero_grad()
            if is_batched_train:
                total_loss = 0
                for batch_x, batch_y, h in zip(X_train, y_train, H_train):
                    x = get_x(batch_x)
                    batch_x_tensor = to_tensor(x)
                    batch_y_tensor = to_tensor(batch_y)
                    h_tensor = to_tensor(h)
                    output = self(batch_x_tensor, h_tensor)
                    loss = self.criterion(output, batch_y_tensor)
                    total_loss += loss
                total_loss.backward()
                return total_loss
            else:
                x = get_x(X_train)
                x_tensor = to_tensor(x)
                y_tensor = to_tensor(y_train)
                h_tensor = to_tensor(H_train)
                output = self(x_tensor, h_tensor)
                loss = self.criterion(output, y_tensor)
                loss.backward()
                return loss

        for epoch in range(n_epochs):
            self.train()

            train_losses = []

            if isinstance(optimizer, optim.LBFGS):
                optimizer.step(closure)
            else:
                if is_batched_train:
                    for batch_x, batch_y, h in zip(X_train, y_train, H_train):
                        x = get_x(batch_x)
                        batch_x_tensor = to_tensor(x)
                        batch_y_tensor = to_tensor(batch_y)
                        h_tensor = to_tensor(h)
                        output = self(batch_x_tensor, h_tensor)
                        loss = self.criterion(output, batch_y_tensor)
                        loss.backward()
                        optimizer.step()
                        train_losses.append(loss.item())
                else:
                    optimizer.zero_grad()
                    x = get_x(X_train)
                    x_tensor = to_tensor(x)
                    y_tensor = to_tensor(y_train)
                    h_tensor = to_tensor(H_train)
                    output = self(x_tensor, h_tensor)
                    loss = self.criterion(output, y_tensor)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

            self.eval()
            val_losses = []
            with torch.no_grad():
                if is_batched_val:
                    for batch_x, batch_y, batch_h in zip(X_val, y_val, H_val):
                        x = get_x(batch_x)
                        batch_x_tensor = to_tensor(x)
                        batch_y_tensor = to_tensor(batch_y)
                        batch_h_tensor = to_tensor(batch_h)
                        output = self(batch_x_tensor, batch_h_tensor)
                        val_loss = self.criterion(output, batch_y_tensor)
                        val_losses.append(val_loss.item())
                else:
                    x = get_x(X_val)
                    x_tensor = to_tensor(x)
                    y_tensor = to_tensor(y_val)
                    h_tensor = to_tensor(H_val)
                    output = self(x_tensor, h_tensor)
                    val_loss = self.criterion(output, y_tensor)
                    val_losses.append(val_loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else closure().item()
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
                f'{self.name}: Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Error: {avg_val_loss:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        if draw_loss:
            plt.ioff()
            plt.show()

        self.load_state_dict(best_model_state)
        return best_val_error

    def get_documentation(self):
        documentation = {"hyperparameters": {
            "optimizer": self.optimizer_class.__name__,
        }}
        return documentation