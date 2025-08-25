from collections import deque
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
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, trapezoid
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sympy.physics.units import acceleration, velocity
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
import Models.model_base as mb

class LinearModel(mb.BaseModel):
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None, name="Linear_Regression"):
        """
        Initializes a Linear Regression model.

        Parameters
        ----------
        fit_intercept : bool, optional
            Whether to calculate the intercept for this model. The default is True.
        normalize : bool, optional
            This parameter is deprecated and will be removed in 1.4. The default is False.
        copy_X : bool, optional
            If True, X will be copied; else, it may be overwritten. The default is True.
        n_jobs : int, optional
            The number of jobs to use for the computation. The default is None.
        """
        self.model = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs)
        self.scaler = None
        # Save Parameter for documentation
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.name = name
        self.device = "cpu"

    def reset_hyperparameter(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
        """
        Reset the hyperparameters of the Linear Regression model.
        """
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        # Reinitialize the model with new hyperparameters
        self.model = LinearRegression(
            fit_intercept=fit_intercept,
            normalize=normalize,
            copy_X=copy_X,
            n_jobs=n_jobs
        )

    def criterion(self, y_target, y_pred):
        """
        Compute the Mean Squared Error (MSE) loss between the target and predicted values.

        Parameters
        ----------
        y_target : array-like
            The target values.
        y_pred : array-like
            The predicted values.

        Returns
        -------
        float
            The computed MSE loss.
        """
        return mean_squared_error(y_target, y_pred)

    def predict(self, X):
        """
        Make predictions based on the input data using the Linear Regression model.

        Parameters
        ----------
        X : array-like
            The input data.

        Returns
        -------
        numpy.ndarray
            The predicted values.
        """
        return self.model.predict(X)

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=1, trial=None, draw_loss=False, n_outlier=12, patience_stop=10, **kwargs):
        """
        Train the Linear Regression model using the training data and validate it using the validation data.

        Parameters
        ----------
        X_train : array-like
            The training input data.
        y_train : array-like
            The training target values.
        X_val : array-like
            The validation input data.
        y_val : array-like
            The validation target values.
        n_epochs : int, optional
            The number of epochs for training. Default is 1 (for compatibility).
        trial : optuna.trial.Trial, optional
            An Optuna trial object used for pruning based on intermediate validation errors.
        draw_loss : bool, optional
            If True, plots training and validation loss after each epoch. Default is False.
        n_outlier: int, optional
            Number of std used to filter out outliers. Default is 12.

        Returns
        -------
        best_val_error : float
            The best validation error achieved during training.
        """
        best_val_error = float('inf')
        if type(X_train) is list:
            X_train = pd.concat(X_train, ignore_index=True)
            y_train = pd.concat(y_train, ignore_index=True)
        if type(X_val) is list:
            X_val = pd.concat(X_val, ignore_index=True)
            y_val = pd.concat(y_val, ignore_index=True)

        # Training loop (for compatibility, though Linear Regression is not iterative)
        self.model.fit(X_train, y_train.squeeze())
        y_val_pred = self.model.predict(X_val)
        val_error = self.criterion(y_val, y_val_pred)

        # Report intermediate values to the pruner
        if trial:
            trial.report(val_error, step=1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Update the best validation error
        if val_error < best_val_error:
            best_val_error = val_error

        if draw_loss:
            plt.plot(1, val_error, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        print(f'{self.name}: Val Error: {val_error:.4f}')
        return best_val_error

    def test_model(self, X, y_target, criterion_test=None):
        """
        Test the model using the test data and compute the loss.

        Parameters
        ----------
        X : array-like
            The test input data.
        y_target : array-like
            The test target values.

        Returns
        -------
        tuple
            A tuple containing the loss and the predicted values.
        """
        if criterion_test is None:
            criterion_test = self.criterion
        y_pred = self.predict(X)
        loss = criterion_test(y_target, y_pred)
        return loss, y_pred

    def get_documentation(self):
        """
        Returns the documentation of the current hyperparameters.
        """
        documentation = {"hyperparameters": {
            "fit_intercept": self.fit_intercept,
            "normalize": self.normalize,
            "copy_X": self.copy_X
        }}
        return documentation

    @staticmethod
    def get_reference_model():
        """
        Returns a reference model with default hyperparameters.
        """
        return LinearModel(fit_intercept=True, normalize=False, copy_X=True)

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
        if self.target_channel == 'curr_x':
            self.axis = 1
        elif self.target_channel == 'curr_y':
            self.axis = 2
        elif self.target_channel == 'curr_z':
            self.axis = 3
        elif self.target_channel == 'curr_sp':
            self.axis = 0
        else:
            throw_error('Pleas select an valid target channel.')

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

        # --- Daten vorbereiten ---
        if not isinstance(X_train, list):
            X_train = [X_train]
            y_train = [y_train]
        if not isinstance(X_val, list):
            X_val = [X_val]
            y_val = [y_val]

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
                        x_tensor = self.get_input_vector_from_df(batch_x)
                        y_tensor = to_tensor(batch_y)
                        output = self.forward(x_tensor)
                        val_loss = self.criterion(output, y_tensor)
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
        self.initial_params = {"a1_init": a1, "a2_init": a2, "a3_init": a3, "b_init": b}
        self.a1 = nn.Parameter(torch.tensor(a1, dtype=torch.float32))
        self.a2 = nn.Parameter(torch.tensor(a2, dtype=torch.float32))
        self.a3 = nn.Parameter(torch.tensor(a3, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(b, dtype=torch.float32))
        self.to(self.device)

    def _initialize(self):
        """Setzt die Parameter auf die initial übergebenen Werte zurück."""
        with torch.no_grad():
            self.a1.copy_(torch.tensor(self.initial_params["a1_init"], dtype=torch.float32))
            self.a2.copy_(torch.tensor(self.initial_params["a2_init"], dtype=torch.float32))
            self.a3.copy_(torch.tensor(self.initial_params["a3_init"], dtype=torch.float32))
            self.b.copy_(torch.tensor(self.initial_params["b_init"], dtype=torch.float32))

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
            "description": "This model combines a linear functions.",
            "parameters": {
                "a1": float(self.a1.detach().cpu().numpy()),
                "a2": float(self.a2.detach().cpu().numpy()),
                "a3": float(self.a3.detach().cpu().numpy()),
                "b": float(self.b.detach().cpu().numpy()),
            }
        }
        return documentation

def get_reference(input_size=None):
    return NaiveModel()

class LuGreModelSciPy(mb.BaseModel):
    def __init__(self,name="LuGre_Model",
                 a1 = 1, a2 = 1,  b = 1,
                 sigma_0=1, sigma_1=1, sigma_2=1,
                 f_s=1, f_c=1, v_s=1,
                 dt = 0.02, target_channel = 'curr_x'):
        self.name = name
        self.a1 = a1
        self.a2 = a2
        self.b = b
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.f_s = f_s
        self.f_c = f_c
        self.v_s = v_s
        self.dt = dt

        self.z = 0.0

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

        acceleration = np.array(acceleration)
        velocity = np.array(velocity)
        force = np.array(force)

        return [acceleration, velocity, force, MRR]

    def criterion(self, y_target, y_pred):
        return np.mean(np.abs(y_target - y_pred))

    @staticmethod
    def stedy_state_equation(X, a1, a2, b, sigma_0, sigma_1, sigma_2, f_s, f_c, v_s, dt=0.02):
        a, v, f = X
        f_friction = f_c * np.sign(v) + (f_s - f_c) * np.exp(-(v / v_s) ** 2) * np.sign(v) + sigma_2 * v
        return a1 * a + a2 * f + f_friction + b

    @staticmethod
    def equation(X, a1, a2, b, sigma_0, sigma_1, sigma_2, f_s, f_c, v_s, dt=0.02):
        acceleration, velocity, force = X

        def g(v):
            eps = 1e-12
            return f_c + (f_s - f_c) * np.exp(-(v / v_s) ** 2) + eps

        def step(v, dt, z):
            dz = v - (sigma_0 * np.abs(v) / g(v)) * z
            z += dz * dt
            f_friction = sigma_0 * z + sigma_1 * dz + sigma_2 * v
            return z, f_friction

        # Initialize z
        z = 0.0
        y = np.zeros_like(velocity, dtype=float)

        # Iterate over each time step
        for i in range(len(velocity)):
            z, f_friction = step(velocity[i], dt, z)
            z = np.clip(z, -1e2, 1e2)
            y[i] = a1 * acceleration[i] + a2 * force[i] + f_friction + b

            # Print progress every 100 time steps
            if (i + 1) % 1000 == 0:
                print(f"Fortschritt: {i + 1}/{len(velocity)} Schritte abgeschlossen.")

        return y

    def predict(self, X):
        [acceleration, velocity, force, MRR] = self.get_input_vector_from_df(X)
        acceleration = acceleration[self.axis].squeeze()
        velocity = velocity[self.axis].squeeze()
        force = force[self.axis].squeeze()

        return self.equation([acceleration, velocity, force], self.a1, self.a2, self.b, self.sigma_0, self.sigma_1, self.sigma_2, self.f_s, self.f_c, self.v_s)

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):

        initial_params = [self.a1, self.a2, self.b, self.sigma_0, self.sigma_1, self.sigma_2, self.f_s, self.f_c, self.v_s]

        [acceleration, velocity, force, MRR] = self.get_input_vector_from_df(X_train)
        acceleration = acceleration[self.axis].squeeze()
        velocity = velocity[self.axis].squeeze()
        force = force[self.axis].squeeze()
        x_data = [acceleration, velocity, force]
        y = np.array(y_train).squeeze()
        params_lugre, y_pred = curve_fit(f = self.equation, xdata = x_data, ydata = y, p0=initial_params, maxfev=10000)

        [self.a1, self.a2, self.b, self.sigma_0, self.sigma_1, self.sigma_2, self.f_s, self.f_c, self.v_s] = params_lugre

        # Ausgabe der trainierten Parameter
        print("Trained Parameters:")
        print(f"a1: {self.a1:.3f}")
        print(f"a2: {self.a2:.3f}")
        print(f"b: {self.b:.3f}")
        print(f"sigma_0: {self.sigma_0:.3f}")
        print(f"sigma_1: {self.sigma_1:.3f}")
        print(f"sigma_2: {self.sigma_2:.3f}")
        print(f"f_s: {self.f_s:.3f}")
        print(f"f_c: {self.f_c:.3f}")
        print(f"v_s: {self.v_s:.3f}")

        validation_loss = self.criterion(y_val.squeeze(), self.predict(X_val))
        print(f"Validation Loss: {validation_loss:.4e}")

        return validation_loss

    def test_model(self, X, y_target):
        prediction = self.predict(X)
        loss = self.criterion(y_target.squeeze(), prediction)
        return loss, prediction

    def get_documentation(self):
        documentation = {
            "description": "This model combines linear functions with the LuGre friction model to simulate friction in dynamic systems. It uses SciPy for curve fitting to train the model parameters.",
            "parameters": {
                "name": self.name,
                "a1": self.a1,
                "a2": self.a2,
                "b": self.b,
                "sigma_0": self.sigma_0,
                "sigma_1": self.sigma_1,
                "sigma_2": self.sigma_2,
                "f_s": self.f_s,
                "f_c": self.f_c,
                "v_s": self.v_s,
                "dt": self.dt,
                "target_channel": self.target_channel
            }
        }
        return documentation

class FrictionModel(mb.BaseModel):
    def __init__(self, name="Friction_Model",
                 f_s = 0, a_x = 0, a_sp = 0, b = 0, f_c = 0, sigma_2 = 0,a_b =0,
                 velocity_threshold=1e-1, target_channel = 'curr_x'):
        self.name = name
        self.velocity_threshold = velocity_threshold
        self.F_s = f_s
        self.theta_f = a_x
        self.a_sp = a_sp
        self.b = b
        self.F_c = f_c
        self.sigma_2 = sigma_2
        self.theta_a = a_b
        self.target_channel = target_channel

    def sign_hold(self, v, eps=1e-1, n=3, init=-1):
        # Initialisierung des Arrays z mit Nullen
        z = np.zeros(len(v))
        h_init = np.ones(n) * init

        assert n > 1

        # Initialisierung des FiFo h mit Länge 5 und Initialwerten 0
        h = deque(h_init, maxlen=n)

        # Berechnung von z
        for i in range(len(v)):
            if abs(v[i]) > eps:
                h.append(v[i])

            if i >= n - 1:  # Da wir ab dem 5. Element starten wollen
                # Berechne zi als Vorzeichen der Summe
                z[i] = np.sign(sum(h))

        return z

    def criterion(self, y_target, y_pred):
        return np.mean(np.abs(y_target - y_pred))

    def predict(self, X):
        for target in self.target_channel:
            axis = target.replace('curr_', '')
            v_x = X[f'v_{axis}_1_current'].values
            a_x = X[f'a_{axis}_1_current'].values
            f_x_sim = X[f'f_{axis}_sim_1_current'].values
            v_sp = X[f'v_sp_1_current'].values

            stillstand_mask = (np.abs(v_x) <= self.velocity_threshold)
            bewegung_mask = ~stillstand_mask

            y_pred = np.zeros_like(v_x, dtype=float)
            v_s = self.sign_hold(v_x)
            y_pred[stillstand_mask] = (self.F_s * v_s[stillstand_mask] +
                                       self.theta_f * f_x_sim[stillstand_mask] +
                                       self.b * v_sp[stillstand_mask])

            y_pred[bewegung_mask] = (self.F_c * np.sign(v_x[bewegung_mask]) +
                                     self.sigma_2 * v_x[bewegung_mask] +
                                     self.theta_f * f_x_sim[bewegung_mask] +
                                     self.theta_a * a_x[bewegung_mask] +
                                     self.b * v_sp[bewegung_mask])

        return y_pred

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        data = X_train.copy()
        data[self.target_channel] = y_train.values

        params, _, _ = self.fit_friction_model(data)

        self.F_s = params['F_s']
        self.theta_f = params['theta_f']
        self.b = params['b']
        self.F_c = params['F_c']
        self.sigma_2 = params['sigma_2']
        self.theta_a = params['theta_a']

        # Ausgabe der Parameter
        for param_name, param_value in params.items():
            print(f"{param_name}: {param_value}")

        validation_loss = self.criterion(y_val.values.squeeze(), self.predict(X_val))
        print(f"Validation Loss: {validation_loss:.4e}")
        return validation_loss

    def fit_friction_model(self, data):
        if type(self.target_channel) is not list:
            self.target_channel = [self.target_channel]
        for target in self.target_channel:
            axis = target.replace('curr_', '')
            v_x = data[f'v_{axis}_1_current'].values
            a_x = data[f'a_{axis}_1_current'].values
            f_x_sim = data[f'f_{axis}_sim_1_current'].values
            v_s = self.sign_hold(v_x)
            v_sp = data[f'v_sp_1_current'].values

            curr_x = data[target].values

            stillstand_mask = (np.abs(v_x) <= self.velocity_threshold)
            bewegung_mask = ~stillstand_mask

            params = {}

            if np.sum(stillstand_mask) > 2:
                X_stillstand = np.column_stack([v_s[stillstand_mask],
                                                f_x_sim[stillstand_mask],
                                                v_sp[stillstand_mask]]) # np.ones(np.sum(stillstand_mask))
                y_stillstand = curr_x[stillstand_mask]
                reg_stillstand = LinearRegression(fit_intercept=False)
                reg_stillstand.fit(X_stillstand, y_stillstand)
                params['F_s'] = reg_stillstand.coef_[0]
                params['theta_f'] = reg_stillstand.coef_[1]
                params['b'] = reg_stillstand.coef_[2]
            else:
                print("Warnung: Nicht genügend Stillstandspunkte für Fitting")
                params['F_s'] = 0
                params['theta_f'] = 1
                params['b'] = 0

            if np.sum(bewegung_mask) > 4:
                X_bewegung = np.column_stack([np.sign(v_x[bewegung_mask]), v_x[bewegung_mask],
                                              f_x_sim[bewegung_mask],
                                              a_x[bewegung_mask], v_sp[bewegung_mask]]) #np.ones(np.sum(bewegung_mask))
                y_bewegung = curr_x[bewegung_mask]
                reg_bewegung = LinearRegression(fit_intercept=False)
                reg_bewegung.fit(X_bewegung, y_bewegung)
                params['F_c'] = reg_bewegung.coef_[0]
                params['sigma_2'] = reg_bewegung.coef_[1]
                params['theta_f'] = reg_bewegung.coef_[2]
                params['theta_a'] = reg_bewegung.coef_[3]
                params['b'] = reg_bewegung.coef_[4]
            else:
                print("Warnung: Nicht genügend Bewegungspunkte für Fitting")
                params['F_c'] = 0
                params['sigma_2'] = 0
                params['theta_f'] = 1
                params['a_y'] = 1
                params['a_z'] = 1
                params['a_sp'] = 0
                params['theta_a'] = 0
                params['b_s'] = 0

        return params, stillstand_mask, bewegung_mask

    def test_model(self, X, y_target):
        prediction = self.predict(X)
        loss = self.criterion(y_target.values.squeeze(), prediction)
        return loss, prediction

    def get_documentation(self):
        documentation = {
            "description": "This model fits a two-stage friction model: stillstand and movement.",
            "parameters": {
                "name": self.name,
                "velocity_threshold": self.velocity_threshold,
                "F_s": self.F_s,
                "a_x": self.theta_f,
                "a_sp": self.a_sp,
                "b": self.b,
                "F_c": self.F_c,
                "sigma_2": self.sigma_2,
                "a_b": self.theta_a,
            }
        }
        return documentation

    def reset_hyperparameter(self):
        throw_error('Not implemented')

class ThermodynamicModel(mb.BaseModel):
    def __init__(self, name="Thermodynamic_Model",
                 theta_v = 0, theta_a = 0, theta_f = 0,
                 velocity_threshold=1e-1, target_channel = 'curr_x'):

        self.name = name
        self.velocity_threshold = velocity_threshold
        self.theta_v = theta_v
        self.theta_f = theta_f
        self.theta_a = theta_a
        self.target_channel = target_channel

    def criterion(self, y_target, y_pred):
        return np.mean(np.abs(y_target - y_pred))

    def predict(self, X):
        for target in self.target_channel:
            axis = target.replace('curr_', '')
            v_x = X[f'v_{axis}_1_current'].values
            a_x = X[f'a_{axis}_1_current'].values
            f_x_sim = X[f'f_{axis}_sim_1_current'].values
            mrr = X[f'materialremoved_sim_1_current'].values
            y_pred = self.theta_v * v_x + self.theta_f * f_x_sim * mrr / v_x + self.theta_a * a_x

        return y_pred

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):

        for target in self.target_channel:
            axis = target.replace('curr_', '')
            v_x = X_train[f'v_{axis}_1_current'].values
            a_x = X_train[f'a_{axis}_1_current'].values
            f_x_sim = X_train[f'f_{axis}_sim_1_current'].values
            mrr = X_train[f'materialremoved_sim_1_current'].values

            params = {}

            data = np.array([a_x, v_x, f_x_sim * mrr / v_x ]).T

            reg = LinearRegression(fit_intercept=False)
            reg.fit(data, y_train[target].values)
            params['theta_a'] = reg.coef_[0]
            params['theta_v'] = reg.coef_[1]
            params['theta_f'] = reg.coef_[2]

        self.theta_v = params['theta_v']
        self.theta_f = params['theta_f']
        self.theta_a = params['theta_a']

        # Ausgabe der Parameter
        for param_name, param_value in params.items():
            print(f"{param_name}: {param_value}")

        validation_loss = self.criterion(y_val.values.squeeze(), self.predict(X_val))
        print(f"Validation Loss: {validation_loss:.4e}")
        return validation_loss


    def test_model(self, X, y_target):
        prediction = self.predict(X)
        loss = self.criterion(y_target.values.squeeze(), prediction)
        return loss, prediction

    def get_documentation(self):
        documentation = {
            "description": "This model fits a two-stage friction model: stillstand and movement.",
            "parameters": {
                "name": self.name,
                "velocity_threshold": self.velocity_threshold,
                "theta_a": self.theta_a,
                "theta_v": self.theta_v,
                "theta_f": self.theta_f,
            }
        }
        return documentation

    def reset_hyperparameter(self):
        throw_error('Not implemented')


class FrictionModelAdvanced(mb.BaseModel):
    def __init__(self, name="Friction_Model", f_s=0, a_x=0, a_y=0, a_z=0, a_sp=0, b=0, f_c=0, sigma_2=0, a_b=0,
                 velocity_threshold=1e-1, target_channel='curr_x'):
        self.name = name
        self.velocity_threshold = velocity_threshold
        self.F_s = f_s
        self.a_x = a_x
        self.a_y = a_y
        self.a_z = a_z
        self.a_sp = a_sp
        self.b = b
        self.F_c = f_c
        self.sigma_2 = sigma_2
        self.a_b = a_b
        self.target_channel = target_channel

    def sign_hold(self, v, eps=1e-1):
        z = np.zeros(len(v))
        h = deque([0, 0, 0, 0, 0], maxlen=5)
        for i in range(len(v)):
            if abs(v[i]) > eps:
                h.append(v[i])
            if i >= 4:
                z[i] = np.sign(sum(h))
        return z

    def criterion(self, y_target, y_pred):
        return np.mean(np.abs(y_target - y_pred))

    def predict(self, X):
        axis = self.target_channel.replace('curr_', '')
        v_x = X[f'v_{axis}_1_current'].values
        a_x = X[f'a_x_1_current'].values
        a_y = X[f'a_y_1_current'].values
        a_z = X[f'a_z_1_current'].values
        a_sp = X[f'a_sp_1_current'].values
        f_x_sim = X[f'f_x_sim_1_current'].values
        f_y_sim = X[f'f_y_sim_1_current'].values
        f_z_sim = X[f'f_z_sim_1_current'].values
        f_sp_sim = X[f'f_sp_sim_1_current'].values

        stillstand_mask = (np.abs(v_x) <= self.velocity_threshold)
        bewegung_mask = ~stillstand_mask
        y_pred = np.zeros_like(v_x, dtype=float)
        v_s = self.sign_hold(v_x)

        y_pred[stillstand_mask] = (self.F_s * v_s[stillstand_mask] +
                                   self.a_x * f_x_sim[stillstand_mask] +
                                   self.a_y * f_y_sim[stillstand_mask] +
                                   self.a_z * f_z_sim[stillstand_mask] +
                                   self.a_sp * f_sp_sim[stillstand_mask] +
                                   self.b)

        y_pred[bewegung_mask] = (self.F_c * np.sign(v_x[bewegung_mask]) +
                                 self.sigma_2 * v_x[bewegung_mask] +
                                 self.a_x * f_x_sim[bewegung_mask] +
                                 self.a_y * f_y_sim[bewegung_mask] +
                                 self.a_z * f_z_sim[bewegung_mask] +
                                 self.a_sp * f_sp_sim[bewegung_mask] +
                                 self.a_b * a_x[bewegung_mask] +
                                 self.b)
        return y_pred

    def train_model(self, X_train, y_train, X_val, y_val, **kwargs):
        data = X_train.copy()
        data[self.target_channel] = y_train.values
        params, _, _ = self.fit_friction_model(data)

        self.F_s = params['F_s']
        self.a_x = params['a_x']
        self.a_y = params['a_y']
        self.a_z = params['a_z']
        self.a_sp = params['a_sp']
        self.b = params['b']
        self.F_c = params['F_c']
        self.sigma_2 = params['sigma_2']
        self.a_b = params['a_b']

        for param_name, param_value in params.items():
            print(f"{param_name}: {param_value}")

        validation_loss = self.criterion(y_val.values.squeeze(), self.predict(X_val))
        print(f"Validation Loss: {validation_loss:.4e}")
        return validation_loss

    def fit_friction_model(self, data):
        axis = self.target_channel.replace('curr_', '')
        v_x = data[f'v_{axis}_1_current'].values
        a_x = data[f'a_x_1_current'].values
        f_x_sim = data[f'f_x_sim_1_current'].values
        f_y_sim = data[f'f_y_sim_1_current'].values
        f_z_sim = data[f'f_z_sim_1_current'].values
        f_sp_sim = data[f'f_sp_sim_1_current'].values

        v_s = self.sign_hold(v_x)
        curr_x = data[self.target_channel].values
        stillstand_mask = (np.abs(v_x) <= self.velocity_threshold)
        bewegung_mask = ~stillstand_mask
        params = {}

        if np.sum(stillstand_mask) > 2:
            X_stillstand = np.column_stack(
                [v_s[stillstand_mask], f_x_sim[stillstand_mask], f_y_sim[stillstand_mask], f_z_sim[stillstand_mask],
                 f_sp_sim[stillstand_mask], np.ones(np.sum(stillstand_mask))])
            y_stillstand = curr_x[stillstand_mask]
            reg_stillstand = LinearRegression(fit_intercept=False)
            reg_stillstand.fit(X_stillstand, y_stillstand)
            params['F_s'] = reg_stillstand.coef_[0]
            params['a_x'] = reg_stillstand.coef_[1]
            params['a_y'] = reg_stillstand.coef_[2]
            params['a_z'] = reg_stillstand.coef_[3]
            params['a_sp'] = reg_stillstand.coef_[4]
            params['b'] = reg_stillstand.coef_[5]
        else:
            print("Warnung: Nicht genügend Stillstandspunkte für Fitting")
            params['F_s'] = 0
            params['a_x'] = 1
            params['a_y'] = 1
            params['a_z'] = 1
            params['a_sp'] = 1
            params['b'] = 0

        if np.sum(bewegung_mask) > 4:
            X_bewegung = np.column_stack(
                [np.sign(v_x[bewegung_mask]), v_x[bewegung_mask], f_x_sim[bewegung_mask], f_y_sim[bewegung_mask],
                 f_z_sim[bewegung_mask], f_sp_sim[bewegung_mask], a_x[bewegung_mask], np.ones(np.sum(bewegung_mask))])
            y_bewegung = curr_x[bewegung_mask]
            reg_bewegung = LinearRegression(fit_intercept=False)
            reg_bewegung.fit(X_bewegung, y_bewegung)
            params['F_c'] = reg_bewegung.coef_[0]
            params['sigma_2'] = reg_bewegung.coef_[1]
            params['a_x'] = reg_bewegung.coef_[2]
            params['a_y'] = reg_bewegung.coef_[3]
            params['a_z'] = reg_bewegung.coef_[4]
            params['a_sp'] = reg_bewegung.coef_[5]
            params['a_b'] = reg_bewegung.coef_[6]
            params['b'] = reg_bewegung.coef_[7]
        else:
            print("Warnung: Nicht genügend Bewegungspunkte für Fitting")
            params['F_c'] = 0
            params['sigma_2'] = 0
            params['a_x'] = 1
            params['a_y'] = 1
            params['a_z'] = 1
            params['a_sp'] = 1
            params['a_b'] = 0
            params['b'] = 0

        return params, stillstand_mask, bewegung_mask

    def test_model(self, X, y_target):
        prediction = self.predict(X)
        loss = self.criterion(y_target.values.squeeze(), prediction)
        return loss, prediction

    def get_documentation(self):
        documentation = {
            "description": "This model fits a two-stage friction model: stillstand and movement.",
            "parameters": {
                "name": self.name,
                "velocity_threshold": self.velocity_threshold,
                "F_s": self.F_s,
                "a_x": self.a_x,
                "a_y": self.a_y,
                "a_z": self.a_z,
                "a_sp": self.a_sp,
                "b": self.b,
                "F_c": self.F_c,
                "sigma_2": self.sigma_2,
                "a_b": self.a_b,
            }
        }
        return documentation

    def reset_hyperparameter(self):
        raise NotImplementedError('Not implemented')