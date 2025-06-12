from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import optuna
from numpy.f2py.auxfuncs import throw_error
from sympy.physics.units import acceleration, velocity
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
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

class BasePhysicalModel(mb.BaseModel, nn.Module, ABC):
    def __init__(self, input_size=None, output_size=1, name="BaseNetModel", learning_rate=1, optimizer_type='quasi_newton', target_channel = 'curr_x'):
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
        super(BasePhysicalModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.target_channel = target_channel
        if target_channel == 'curr_x':
            self.axis = 1
        elif target_channel == 'curr_y':
            self.axis = 2
        elif target_channel == 'curr_z':
            self.axis = 3
        elif target_channel == 'curr_sp':
            self.axis = 0
        else:
            throw_error('Pleas select an valid target channel.')

    def criterion(self, y_target, y_pred):
        criterion = nn.MSELoss()
        return criterion(y_target.squeeze(), y_pred.squeeze())

    def predict(self, X):
        if type(X) is pd.DataFrame:
            X = X.values
        with torch.no_grad():
            return self.forward(X)

    def get_input_vector_from_df(self, df):
        prefix_current = '_1_current'
        axes = ['sp','x', 'y', 'z']
        acceleration = []
        velocity = []
        force = []
        for axis in axes:
            acceleration.append(df[['a_' + axis + prefix_current]].values)
            velocity.append(df[['v_' + axis + prefix_current]].values)
            force.append(df[['f_' + axis + '_sim' + prefix_current]].values)
        MRR = df['materialremoved_sim' + prefix_current].values

        acceleration = torch.tensor(np.array(acceleration), dtype=torch.float32).squeeze().to(self.device)
        velocity = torch.tensor(np.array(velocity), dtype=torch.float32).squeeze().to(self.device)
        force = torch.tensor(np.array(force), dtype=torch.float32).squeeze().to(self.device)
        MRR = torch.tensor(MRR, dtype=torch.float32).squeeze().to(self.device)
        return [acceleration, velocity, force, MRR]

    def get_input_vector_from_tensor(self, input_vector):
        if type(input_vector) is list:
            [acceleration, velocity, force, MRR] = input_vector # shape: [4, T] je Achse
        elif type(input_vector) is torch.Tensor:
            if input_vector.shape[1] == 13:
                input_vector = input_vector.T
                acceleration = [input_vector[0,:], input_vector[1,:], input_vector[2,:], input_vector[3,:]]
                force = [input_vector[4, :], input_vector[5, :], input_vector[6, :], input_vector[7, :]]
                MRR = input_vector[8, :]
                velocity = [input_vector[9,:], input_vector[10,:], input_vector[11,:], input_vector[12,:]]
            elif input_vector.shape[1] == 4:
                [acceleration, force, MRR, velocity] = input_vector.T
            else:
                throw_error(f'input_vector shape {input_vector.shape}, wrong shape')
        else:
            throw_error(f'input_vector is of type {type(input_vector)} but should be of type list or torch.tensor')

        return acceleration, velocity, force, MRR

    def test_model(self, X, y_target, criterion_test=None):
        if criterion_test is None:
            criterion_test = self.criterion
        input_vector = self.get_input_vector_from_df(X)
        if not isinstance(y_target, np.ndarray):
            y_target = y_target.to_numpy()
        y_target = torch.tensor(y_target, dtype=torch.float32).to(self.device)
        y_pred = self.predict(input_vector)
        loss = criterion_test(y_target, y_pred)
        return loss.item(), y_pred.detach().cpu().numpy()

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=100, patience=10, draw_loss=False, epsilon=0.00005,
                    trial=None, n_outlier=12, reset_parameters=True):
        if reset_parameters:
            print(f"{self.name}: Setze Parameter auf Initialwerte zurück.")
            self._initialize()

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
        assert (not is_batched_val) or (len(X_val) == len(y_val)), "Validierungslisten must have the same length"

        print(f"Device: {self.device} | Batched: {is_batched_train}")

        if self.optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type.lower() == 'quasi_newton':
            optimizer = optim.LBFGS(self.parameters(), lr=self.learning_rate, line_search_fn='strong_wolfe') #  max_iter=20, history_size=10,
            patience = 4
        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

        if patience < 4:
            patience = 4
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                               patience=int(patience / 2))

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

            def closure():
                optimizer.zero_grad()
                if is_batched_train:
                    loss_total = 0
                    for batch_x, batch_y in zip(X_train, y_train):
                        batch_x_tensor = self.get_input_vector_from_df(batch_x)
                        batch_y_tensor = to_tensor(batch_y)
                        output = self.forward(batch_x_tensor)
                        loss = self.criterion(batch_y_tensor, output)
                        loss.backward()
                        loss_total += loss
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    return loss_total
                else:
                    x_tensor = self.get_input_vector_from_df(X_train)
                    y_tensor = to_tensor(y_train)
                    output = self.forward(x_tensor)
                    loss = self.criterion(y_tensor, output)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    return loss

            if self.optimizer_type.lower() == 'quasi_newton':
                loss = optimizer.step(closure)
                train_losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
            else:
                if is_batched_train:
                    for batch_x, batch_y in zip(X_train, y_train):
                        optimizer.zero_grad()
                        batch_x_tensor = self.get_input_vector_from_df(batch_x)
                        batch_y_tensor = to_tensor(batch_y)
                        output = self.forward(batch_x_tensor)
                        loss = self.criterion(output, batch_y_tensor)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        optimizer.step()
                        train_losses.append(loss.item())
                else:
                    optimizer.zero_grad()
                    x_tensor = self.get_input_vector_from_df(X_train)
                    y_tensor = to_tensor(y_train)
                    output = self.forward(x_tensor)
                    loss = self.criterion(output, y_tensor)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(loss.item())

            self.eval()
            val_losses = []
            with torch.no_grad():
                if is_batched_val:
                    for batch_x, batch_y in zip(X_val, y_val):
                        batch_x_tensor = self.get_input_vector_from_df(batch_x)
                        batch_y_tensor = to_tensor(batch_y)
                        output = self.forward(batch_x_tensor)
                        val_loss = self.criterion(output, batch_y_tensor)
                        val_losses.append(val_loss.item())
                else:
                    x_tensor = self.get_input_vector_from_df(X_val)
                    y_tensor = to_tensor(y_val)
                    output = self.forward(x_tensor)
                    val_loss = self.criterion(output, y_tensor)
                    val_losses.append(val_loss.item())
            del x_tensor, y_tensor, output, loss
            torch.cuda.empty_cache()

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)

            if avg_val_loss < best_val_error - epsilon:
                best_val_error = avg_val_loss
                best_model_state = self.state_dict()
                patience_counter = 0
            elif epoch > (n_epochs / 10) or self.optimizer_type.lower() == 'quasi_newton':
                patience_counter += 1

            scheduler.step(avg_val_loss)

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

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
                f'{self.name}: Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.4f} Val Error: {avg_val_loss:.4f}, Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        if draw_loss:
            plt.ioff()
            plt.show()

        self.load_state_dict(best_model_state)

        print("Beste Parameter:")
        for name, param in self.named_parameters():
            print(f"{name}: {param.data.cpu().numpy()}")

        return best_val_error

class PhysicalModelErd(BasePhysicalModel):
    def __init__(self, learning_rate=1, optimizer_type='quasi_newton', name="Physical_Model"):
        super(PhysicalModelErd, self).__init__()
        self.initial_params = {}
        # Kopplungsgewichte θ_{i,j}, wobei i ≠ j (Diagonale = 0)
        self.theta = nn.Parameter(torch.zeros(4, 5, dtype=torch.float32))  # 4 Achsen: x,y,z,sp
        self.initial_params["theta"] = self.theta.detach().cpu().numpy().copy()
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _initialize(self):
        """Setzt die Parameter auf die initial übergebenen Werte zurück."""
        with torch.no_grad():
            self.theta.copy_(torch.tensor(self.initial_params["theta"], dtype=torch.float32))

    def forward(self, input_vector):
        [acceleration, velocity, force, MRR] = input_vector  # shape: [4, T] je Achse

        movment =  self.theta[:, 0] * (acceleration * velocity).T + self.theta[:, 1]  * (velocity ** 2 * torch.sign(velocity)).T
        material = self.theta[:, 2]  * (force * MRR).T
        current = movment + material + self.theta[:, 3]  * torch.sign(velocity).T + self.theta[:, 4] # shape: [4, T]

        return torch.sum(current, axis=1).unsqueeze(-1)
    def get_documentation(self):
        # θ-Matrix als Dictionary mit Schlüssel wie "theta_0_1": Wert

        theta_dict = {
            f"theta_{i}_{j}": self.theta[i, j].item()
            for i in range(4)
            for j in range(4)
            if i != j  # Diagonale überspringen (optional)
        }

        documentation = {
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "optimizer_type": self.optimizer_type,
                'target_channel': self.target_channel,
                **theta_dict
            }
        }
        return documentation

class PhysicalModelErdSingleAxis(BasePhysicalModel):
    def __init__(self, *args, c_1=1e-4, c_2=1e-3, c_3=1e-7, c_4=1e-1, c_5=1e-3, learning_rate=1, optimizer_type='quasi_newton',
                 name="Physical_Model_Single_Axis",  **kwargs):
        super(PhysicalModelErdSingleAxis, self).__init__(*args, **kwargs)
        self.initial_params = {"c_1": c_1, "c_2": c_2, "c_3": c_3, "c_4": c_4, "c_5": c_5}
        self.c_1 = nn.Parameter(torch.tensor(c_1, dtype=torch.float32))
        self.c_2 = nn.Parameter(torch.tensor(c_2, dtype=torch.float32))
        self.c_3 = nn.Parameter(torch.tensor(c_3, dtype=torch.float32))
        self.c_4 = nn.Parameter(torch.tensor(c_4, dtype=torch.float32))
        self.c_5 = nn.Parameter(torch.tensor(c_5, dtype=torch.float32))

        # Kopplungsgewichte θ_{i,j}, wobei i ≠ j (Diagonale = 0)
        self.theta = nn.Parameter(torch.zeros(4, 5, dtype=torch.float32))  # 4 Achsen: x,y,z,sp
        self.initial_params["theta"] = self.theta.detach().cpu().numpy().copy()
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _initialize(self):
        """Setzt die Parameter auf die initial übergebenen Werte zurück."""
        with torch.no_grad():
            self.c_1.copy_(torch.tensor(self.initial_params["c_1"], dtype=torch.float32))
            self.c_2.copy_(torch.tensor(self.initial_params["c_2"], dtype=torch.float32))
            self.c_3.copy_(torch.tensor(self.initial_params["c_3"], dtype=torch.float32))
            self.c_4.copy_(torch.tensor(self.initial_params["c_4"], dtype=torch.float32))
            self.c_5.copy_(torch.tensor(self.initial_params["c_5"], dtype=torch.float32))

    def forward(self, input_vector):
        acceleration, velocity, force, MRR = self.get_input_vector_from_tensor(input_vector)
        if self.axis is not None and len(acceleration) > 1:
            acceleration = acceleration[self.axis]
            velocity = velocity[self.axis]
            force = force[self.axis]
        movment = self.c_1 * acceleration * velocity + self.c_2 * velocity ** 2 * torch.sign(velocity)
        material = self.c_3 * force * MRR
        current = movment + material + self.c_4 * torch.sign(velocity) + self.c_5 # shape: [4, T]

        current = current.T
        return current.unsqueeze(-1) # Output dimensions of NN

    def get_documentation(self):
        # θ-Matrix als Dictionary mit Schlüssel wie "theta_0_1": Wert
        """
        theta_dict = {
            f"theta_{i}_{j}": self.theta[i, j].item()
            for i in range(4)
            for j in range(4)
            if i != j  # Diagonale überspringen (optional)
        }
        """
        documentation = {
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "optimizer_type": self.optimizer_type,
                "c_1": self.c_1.item(),
                "c_2": self.c_2.item(),
                "c_3": self.c_3.item(),
                "c_4": self.c_4.item(),
                "c_5": self.c_5.item(),
            }
        }
        return documentation

class NaiveModelSigmoid(BasePhysicalModel):
    def __init__(self, *args, name="Naive_Model_Sigmoid", learning_rate=1,
                 optimizer_type='quasi_newton', a1 = -1, b1 = 0, a2 = -1, b2 = 0, c2 = 0, a3 = -1,  **kwargs):
        super(NaiveModelSigmoid, self).__init__(*args, name=name, learning_rate=learning_rate, optimizer_type=optimizer_type, **kwargs)

        self.initial_params = {"a1": a1, "b1": b1, "a2": a2, "b2": b2, "c2": c2, "a3": a3}
        self.a1 = nn.Parameter(torch.tensor(a1, dtype=torch.float32))
        self.b1 = nn.Parameter(torch.tensor(b1, dtype=torch.float32))
        self.a2 =  nn.Parameter(torch.tensor(a2, dtype=torch.float32))
        self.b2 = nn.Parameter(torch.tensor(b2, dtype=torch.float32))
        self.c2 = nn.Parameter(torch.tensor(c2, dtype=torch.float32))
        self.a3 = nn.Parameter(torch.tensor(a3, dtype=torch.float32))

        self.to(self.device)

    def _initialize(self):
        """Setzt die Parameter auf die initial übergebenen Werte zurück."""
        with torch.no_grad():
            self.a1.copy_(torch.tensor(self.initial_params["a1"], dtype=torch.float32))
            self.b1.copy_(torch.tensor(self.initial_params["b1"], dtype=torch.float32))
            self.a2.copy_(torch.tensor(self.initial_params["a2"], dtype=torch.float32))
            self.b2.copy_(torch.tensor(self.initial_params["b2"], dtype=torch.float32))
            self.c2.copy_(torch.tensor(self.initial_params["c2"], dtype=torch.float32))
            self.a3.copy_(torch.tensor(self.initial_params["a3"], dtype=torch.float32))

    def sigmoid_stable(self, x):
        """Verwendet die eingebaute, numerisch stabile Sigmoid-Funktion von PyTorch."""
        # Sicherstellen, dass x ein Tensor ist
        if isinstance(x, list):
            x = torch.stack(x) if len(x) > 1 else x[0]

        # Verschieben und Clipping
        x_shifted = x + self.b2
        x_clipped = torch.clamp(x_shifted, min=-50, max=50)

        # PyTorch's eingebaute Sigmoid-Funktion verwenden
        sigmoid_part = torch.sigmoid(x_clipped) * (self.a2) + self.c2

        # Finales Clipping
        result = torch.clamp(sigmoid_part, min=-1e6, max=1e6)

        return result

    def forward(self, x):
        acceleration, velocity, force, MRR = self.get_input_vector_from_tensor(x)

        # Sicherstellen, dass die Eingaben Tensors sind
        force_x = force[self.axis] if isinstance(force[self.axis], torch.Tensor) else torch.tensor(force[self.axis], dtype=torch.float32,
                                                                                   device=self.device)
        velocity_x = velocity[self.axis] if isinstance(velocity[self.axis], torch.Tensor) else torch.tensor(velocity[self.axis],
                                                                                            dtype=torch.float32,
                                                                                            device=self.device)
        acceleration_x = acceleration[self.axis] if isinstance(acceleration[self.axis], torch.Tensor) else torch.tensor(acceleration[self.axis],
                                                                                            dtype=torch.float32,
                                                                                            device=self.device)

        y_force = self.a1 * force_x + self.b1
        y_acceleration = self.a3 * acceleration_x

        # Sigmoid Teil mit noch mehr Schutz
        y_v = self.sigmoid_stable(velocity_x)

        # Finale Addition mit Schutz
        result = y_force + y_acceleration + y_v
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("ERROR: Final result has NaN/Inf!")
            result = torch.clamp(result, min=-1e6, max=1e6)
            result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

        return result

    def criterion(self, y_target, y_pred):
        """Robuste Loss-Funktion mit NaN-Handling"""
        # Shapes angleichen
        y_target = y_target.squeeze()
        y_pred = y_pred.squeeze()

        # Debug: Shapes prüfen
        if y_target.shape != y_pred.shape:
            print(f"Shape mismatch: y_target {y_target.shape}, y_pred {y_pred.shape}")
            # Kleinere Dimension erweitern
            if y_target.dim() < y_pred.dim():
                y_target = y_target.unsqueeze(-1)
            elif y_pred.dim() < y_target.dim():
                y_pred = y_pred.unsqueeze(-1)

        # NaN-Werte identifizieren (elementweise)
        nan_mask_pred = torch.isnan(y_pred)
        nan_mask_target = torch.isnan(y_target)
        mask = ~(nan_mask_pred | nan_mask_target)

        # Prüfen ob überhaupt gültige Werte vorhanden sind
        if mask.sum() == 0:
            print("Warning: All values are NaN!")
            return torch.tensor(1e6, requires_grad=True, device=self.device)

        # Nur gültige Werte verwenden
        y_pred_clean = y_pred[mask]
        y_target_clean = y_target[mask]

        # MSE Loss berechnen
        mse_loss = nn.MSELoss()(y_target_clean, y_pred_clean)

        # Extreme Losses clippen
        if mse_loss > 1e6:
            mse_loss = torch.tensor(1e6, requires_grad=True, device=self.device)

        return mse_loss

    def get_documentation(self):
        documentation = {
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "optimizer_type": self.optimizer_type,
                'target_channel': self.target_channel,
                **self.initial_params,
            },
            "description": "This model combines a linear and a numerically stable sigmoid function."
        }
        return documentation

class NaiveModel(BasePhysicalModel):
    def __init__(self,*args, name="Naive_Model", learning_rate=1,
                 optimizer_type='quasi_newton', a1 = -1e-3, a2 = -1e-3,  a3 = -1e-1, b = 1e-2, **kwargs):
        super(NaiveModel, self).__init__(*args, name = name, learning_rate=learning_rate, optimizer_type=optimizer_type, **kwargs)

        # VIEL kleinere und sicherere Initialwerte
        self.initial_params = {"a1": a1, "a2": a2, "a3": a3, "b": b}
        self.a1 = nn.Parameter(torch.tensor(a1, dtype=torch.float32))
        self.a2 = nn.Parameter(torch.tensor(a2, dtype=torch.float32))
        self.a3 = nn.Parameter(torch.tensor(a3, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

        self.to(self.device)

    def _initialize(self):
        """Setzt die Parameter auf die initial übergebenen Werte zurück."""
        with torch.no_grad():
            self.a1.copy_(torch.tensor(self.initial_params["a1"], dtype=torch.float32))
            self.a2.copy_(torch.tensor(self.initial_params["a2"], dtype=torch.float32))
            self.a3.copy_(torch.tensor(self.initial_params["a3"], dtype=torch.float32))
            self.b.copy_(torch.tensor(self.initial_params["b"], dtype=torch.float32))

    def forward(self, x):
        acceleration, velocity, force, MRR = self.get_input_vector_from_tensor(x)

        # Sicherstellen, dass die Eingaben Tensors sind
        force_x = force[self.axis] if isinstance(force[self.axis], torch.Tensor) else torch.tensor(force[self.axis], dtype=torch.float32,
                                                                                   device=self.device)
        velocity_x = velocity[self.axis] if isinstance(velocity[self.axis], torch.Tensor) else torch.tensor(velocity[self.axis],
                                                                                            dtype=torch.float32,
                                                                                            device=self.device)
        acceleration_x = acceleration[self.axis] if isinstance(acceleration[self.axis], torch.Tensor) else torch.tensor(acceleration[self.axis],
                                                                                            dtype=torch.float32,
                                                                                            device=self.device)
        # NaN-Checks für Inputs
        if torch.isnan(force_x).any():
            print("ERROR: NaN in force_y input!")
            return torch.full_like(force_x, 0.0)
        if torch.isnan(velocity_x).any():
            print("ERROR: NaN in velocity_y input!")
            return torch.full_like(velocity_x, 0.0)

        # Parameter-Checks
        if torch.isnan(self.a1) or torch.isnan(self.a3):
            print("ERROR: NaN in linear parameters!")
            return torch.full_like(force_x, 0.0)

        if torch.isnan(self.a2) or torch.isnan(self.b):
            print("ERROR: NaN in sigmoid parameters!")
            return torch.full_like(force_x, 0.0)

        y_force = self.a1 * force_x
        y_acceleration = self.a2 * acceleration_x
        y_v = self.a3 * torch.sign(velocity_x)

        # Finale Addition mit Schutz
        result = y_force + y_acceleration + y_v + self.b
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("ERROR: Final result has NaN/Inf!")
            result = torch.clamp(result, min=-1e6, max=1e6)
            result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

        return result

    def criterion(self, y_target, y_pred):
        """Robuste Loss-Funktion mit NaN-Handling"""
        # Shapes angleichen
        y_target = y_target.squeeze()
        y_pred = y_pred.squeeze()

        # Debug: Shapes prüfen
        if y_target.shape != y_pred.shape:
            print(f"Shape mismatch: y_target {y_target.shape}, y_pred {y_pred.shape}")
            # Kleinere Dimension erweitern
            if y_target.dim() < y_pred.dim():
                y_target = y_target.unsqueeze(-1)
            elif y_pred.dim() < y_target.dim():
                y_pred = y_pred.unsqueeze(-1)

        # NaN-Werte identifizieren (elementweise)
        nan_mask_pred = torch.isnan(y_pred)
        nan_mask_target = torch.isnan(y_target)
        mask = ~(nan_mask_pred | nan_mask_target)

        # Prüfen ob überhaupt gültige Werte vorhanden sind
        if mask.sum() == 0:
            print("Warning: All values are NaN!")
            return torch.tensor(1e6, requires_grad=True, device=self.device)

        # Nur gültige Werte verwenden
        y_pred_clean = y_pred[mask]
        y_target_clean = y_target[mask]

        # MSE Loss berechnen
        mse_loss = nn.MSELoss()(y_target_clean, y_pred_clean)

        # Extreme Losses clippen
        if mse_loss > 1e6:
            mse_loss = torch.tensor(1e6, requires_grad=True, device=self.device)

        return mse_loss

    def get_documentation(self):
        documentation = {
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "optimizer_type": self.optimizer_type,
                'target_channel': self.target_channel,
                **self.initial_params,
            },
            "description": "This model combines a linear functions."
        }
        return documentation

class NaiveModel2(BasePhysicalModel):
    def __init__(self,*args, name="Naive_Model", learning_rate=1,
                 optimizer_type='quasi_newton', a1 = -1e-3, a2 = -1e-3,  a3 = -1e-1, b = 1e-2, **kwargs):
        super(NaiveModel2, self).__init__(*args, name = name, learning_rate=learning_rate, optimizer_type=optimizer_type, **kwargs)

        # VIEL kleinere und sicherere Initialwerte
        self.initial_params = {"a1": a1, "a2": a2, "a3": a3, "b": b}
        self.a1 = nn.Parameter(torch.tensor(a1, dtype=torch.float32))
        self.a2 = nn.Parameter(torch.tensor(a2, dtype=torch.float32))
        self.a3 = nn.Parameter(torch.tensor(a3, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))

        self.to(self.device)

    def _initialize(self):
        """Setzt die Parameter auf die initial übergebenen Werte zurück."""
        with torch.no_grad():
            self.a1.copy_(torch.tensor(self.initial_params["a1"], dtype=torch.float32))
            self.a2.copy_(torch.tensor(self.initial_params["a2"], dtype=torch.float32))
            self.a3.copy_(torch.tensor(self.initial_params["a3"], dtype=torch.float32))
            self.b.copy_(torch.tensor(self.initial_params["b"], dtype=torch.float32))

    def get_input_vector_from_tensor(self, input_vector):
        if type(input_vector) is list:
            [acceleration, velocity, force, MRR] = input_vector # shape: [4, T] je Achse
        elif type(input_vector) is torch.Tensor:
            if input_vector.shape[1] == 14:
                input_vector = input_vector.T
                acceleration = [input_vector[0,:], input_vector[1,:], input_vector[2,:], input_vector[3,:]]
                force = [input_vector[4, :], input_vector[5, :], input_vector[6, :], input_vector[7, :]]
                MRR = input_vector[8, :]
                velocity = [input_vector[9,:], input_vector[10,:], input_vector[11,:], input_vector[12,:]]
                y = input_vector[13, :]
            elif input_vector.shape[1] == 5:  # Wenn mit NaiveModel-Prediction
                [acceleration, force, MRR, velocity, y] = input_vector.T
            else:
                throw_error(f'input_vector shape {input_vector.shape}, wrong shape')
        else:
            throw_error(f'input_vector is of type {type(input_vector)} but should be of type list or torch.tensor')

        return acceleration, velocity, force, MRR, y

    def forward(self, x):
        acceleration, velocity, force, MRR, y = self.get_input_vector_from_tensor(x)

        # Sicherstellen, dass die Eingaben Tensors sind
        force_x = force[self.axis] if isinstance(force[self.axis], torch.Tensor) else torch.tensor(force[self.axis], dtype=torch.float32,
                                                                                   device=self.device)
        velocity_x = velocity[self.axis] if isinstance(velocity[self.axis], torch.Tensor) else torch.tensor(velocity[self.axis],
                                                                                            dtype=torch.float32,
                                                                                            device=self.device)
        acceleration_x = acceleration[self.axis] if isinstance(acceleration[self.axis], torch.Tensor) else torch.tensor(acceleration[self.axis],
                                                                                            dtype=torch.float32,
                                                                                            device=self.device)
        # NaN-Checks für Inputs
        if torch.isnan(force_x).any():
            print("ERROR: NaN in force_y input!")
            return torch.full_like(force_x, 0.0)
        if torch.isnan(velocity_x).any():
            print("ERROR: NaN in velocity_y input!")
            return torch.full_like(velocity_x, 0.0)

        # Parameter-Checks
        if torch.isnan(self.a1) or torch.isnan(self.a3):
            print("ERROR: NaN in linear parameters!")
            return torch.full_like(force_x, 0.0)

        if torch.isnan(self.a2) or torch.isnan(self.b):
            print("ERROR: NaN in sigmoid parameters!")
            return torch.full_like(force_x, 0.0)

        y_force = self.a1 * force_x
        y_acceleration = self.a2 * acceleration_x
        y_v = self.a3 * y

        # Finale Addition mit Schutz
        result = y_force + y_acceleration + y_v + self.b
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("ERROR: Final result has NaN/Inf!")
            result = torch.clamp(result, min=-1e6, max=1e6)
            result = torch.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

        return result

    def criterion(self, y_target, y_pred):
        """Robuste Loss-Funktion mit NaN-Handling"""
        # Shapes angleichen
        y_target = y_target.squeeze()
        y_pred = y_pred.squeeze()

        # Debug: Shapes prüfen
        if y_target.shape != y_pred.shape:
            print(f"Shape mismatch: y_target {y_target.shape}, y_pred {y_pred.shape}")
            # Kleinere Dimension erweitern
            if y_target.dim() < y_pred.dim():
                y_target = y_target.unsqueeze(-1)
            elif y_pred.dim() < y_target.dim():
                y_pred = y_pred.unsqueeze(-1)

        # NaN-Werte identifizieren (elementweise)
        nan_mask_pred = torch.isnan(y_pred)
        nan_mask_target = torch.isnan(y_target)
        mask = ~(nan_mask_pred | nan_mask_target)

        # Prüfen ob überhaupt gültige Werte vorhanden sind
        if mask.sum() == 0:
            print("Warning: All values are NaN!")
            return torch.tensor(1e6, requires_grad=True, device=self.device)

        # Nur gültige Werte verwenden
        y_pred_clean = y_pred[mask]
        y_target_clean = y_target[mask]

        # MSE Loss berechnen
        mse_loss = nn.MSELoss()(y_target_clean, y_pred_clean)

        # Extreme Losses clippen
        if mse_loss > 1e6:
            mse_loss = torch.tensor(1e6, requires_grad=True, device=self.device)

        return mse_loss

    def get_documentation(self):
        documentation = {
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "optimizer_type": self.optimizer_type,
                'target_channel': self.target_channel,
                **self.initial_params,
            },
            "description": "This model combines a linear functions."
        }
        return documentation