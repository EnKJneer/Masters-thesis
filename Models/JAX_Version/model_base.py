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
import Models.model_base as mb

class BaseNetModel(mb.BaseModel):
    def __init__(self, input_size=None, output_size=1, name="BaseNetModel", learning_rate=0.001, optimizer_type='adam'):
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
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate  # Store initial value
        self.optimizer_type = optimizer_type
        self.scaler = None
        self.params = None
        self.key = random.PRNGKey(42)

        # JIT compile functions for better performance
        self._jit_forward = None
        self._jit_loss = None
        self._jit_update = None

    @abstractmethod
    def _initialize(self):
        pass

    @abstractmethod
    def forward(self, params, x):
        pass

    def criterion(self, y_target, y_pred, delta=0.22):
        """Huber loss implementation in JAX - matching PyTorch HuberLoss exactly"""
        # PyTorch HuberLoss: pred - target, not target - pred
        diff = y_pred.squeeze() - y_target.squeeze()
        is_small_error = jnp.abs(diff) <= delta
        squared_loss = 0.5 * diff ** 2
        linear_loss = delta * jnp.abs(diff) - 0.5 * delta ** 2
        return jnp.mean(jnp.where(is_small_error, squared_loss, linear_loss))

    def predict(self, X):
        if not isinstance(X, jnp.ndarray):
            X = self.scaled_to_tensor(X)
        return self._jit_forward(self.params, X)

    def scale_data(self, X):
        """
        Scale the input data using the scaler fitted during training.

        Parameters:
        X (array-like): The input data.

        Returns:
        ndarray: The scaled input data.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        return self.scaler.transform(X)

    def test_model(self, X, y_target, criterion_test=None):
        if criterion_test is None:
            criterion_test = self.criterion
        X_scaled = self.scale_data(X)
        X = jnp.array(X_scaled, dtype=jnp.float32)
        if not isinstance(y_target, np.ndarray):
            y_target = y_target.to_numpy()
        y_target = jnp.array(y_target, dtype=jnp.float32)
        y_pred = self.predict(X)
        loss = criterion_test(y_target, y_pred)
        return float(loss), np.array(y_pred)

    def scaled_to_tensor(self, data):
        if isinstance(data, jnp.ndarray):
            return data
        elif hasattr(data, 'values'):
            data_scaled = self.scale_data(data.values)
            return jnp.array(data_scaled, dtype=jnp.float32)
        else:
            # Falls numpy array oder anderes
            data_scaled = self.scale_data(data)
            return jnp.array(data_scaled, dtype=jnp.float32)

    def to_tensor(self, data):
        if isinstance(data, jnp.ndarray):
            return data
        elif hasattr(data, 'values'):
            return jnp.array(data.values, dtype=jnp.float32)
        else:
            # Falls numpy array oder anderes
            return jnp.array(data, dtype=jnp.float32)

    def _setup_optimizer(self):
        """Setup optimizer based on optimizer_type"""
        if self.optimizer_type.lower() == 'adam':
            return optax.adam(self.learning_rate)
        elif self.optimizer_type.lower() == 'quasi_newton':
            # Use Adam with same learning rate for consistency
            return optax.adam(self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

    def _loss_fn(self, params, x, y):
        """Loss function for training"""
        pred = self.forward(params, x)
        return self.criterion(y, pred)

    def _update_step(self, params, opt_state, x, y):
        """Single optimization step"""
        loss, grads = jax.value_and_grad(self._loss_fn)(params, x, y)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=100, patience=10, draw_loss=False, epsilon=0.00005,
                    trial=None, n_outlier=12, reset_parameters=True):

        # Prüfen, ob Inputs Listen sind
        is_batched_train = isinstance(X_train, list) and isinstance(y_train, list)
        is_batched_val = isinstance(X_val, list) and isinstance(y_val, list)

        assert (not is_batched_train) or (len(X_train) == len(y_train)), "Trainingslist must have the same length"
        assert (not is_batched_val) or (len(X_val) == len(y_val)), "Validierungslisten must have the same length"

        print(f"Device: JAX | Batched: {is_batched_train}")

        # Needed for initialization of the layer
        flag_initialization = False
        if self.input_size is None:
            # Define input size and set flag for initialization
            if is_batched_train:
                self.input_size = X_train[0].shape[1]
            else:
                self.input_size = X_train.shape[1]
            flag_initialization = True

        if flag_initialization or reset_parameters or self.params is None:
            self._initialize()

        # Reset learning rate to initial value at start of training
        self.learning_rate = self.initial_learning_rate

        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        opt_state = self.optimizer.init(self.params)

        # Setup JIT compiled functions
        self._jit_forward = jit(self.forward)
        self._jit_loss = jit(self._loss_fn)
        self._jit_update = jit(self._update_step)

        # Scheduler simulation (reduce learning rate on plateau)
        scheduler_patience = 3
        scheduler_factor = 0.5
        scheduler_counter = 0
        best_scheduler_loss = float('inf')

        if patience < 4:
            patience = 4

        if draw_loss:
            loss_vals, epochs, loss_train = [], [], []
            fig, ax = plt.subplots()
            line_val, = ax.plot(epochs, loss_vals, 'r-', label='validation')
            line_train, = ax.plot(epochs, loss_train, 'b-', label='training')
            ax.legend()

        best_val_error = float('inf')
        patience_counter = 0
        best_params = None

        for epoch in range(n_epochs):
            train_losses = []

            # Training step
            if is_batched_train:
                for batch_x, batch_y in zip(X_train, y_train):
                    batch_x_tensor = self.scaled_to_tensor(batch_x)
                    batch_y_tensor = self.to_tensor(batch_y)
                    # Fix: Use same argument order as PyTorch version
                    self.params, opt_state, loss = self._jit_update(self.params, opt_state, batch_x_tensor,
                                                                    batch_y_tensor)
                    train_losses.append(float(loss))
            else:
                x_tensor = self.scaled_to_tensor(X_train)
                y_tensor = self.to_tensor(y_train)
                self.params, opt_state, loss = self._jit_update(self.params, opt_state, x_tensor, y_tensor)
                train_losses.append(float(loss))

            # Validation step
            val_losses = []
            if is_batched_val:
                for batch_x, batch_y in zip(X_val, y_val):
                    batch_x_tensor = self.scaled_to_tensor(batch_x)
                    batch_y_tensor = self.to_tensor(batch_y)
                    val_loss = self._jit_loss(self.params, batch_x_tensor, batch_y_tensor)
                    val_losses.append(float(val_loss))
            else:
                x_tensor = self.scaled_to_tensor(X_val)
                y_tensor = self.to_tensor(y_val)
                val_loss = self._jit_loss(self.params, x_tensor, y_tensor)
                val_losses.append(float(val_loss))

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_val_loss = sum(val_losses) / len(val_losses)

            # Early stopping logic
            if avg_val_loss < best_val_error - epsilon:
                best_val_error = avg_val_loss
                best_params = jax.tree.map(lambda x: x.copy(), self.params)
                patience_counter = 0
            elif epoch > (n_epochs / 10) and epoch > 10:
                patience_counter += 1

            # Scheduler logic (simulate ReduceLROnPlateau)
            if avg_val_loss < best_scheduler_loss - epsilon:
                best_scheduler_loss = avg_val_loss
                scheduler_counter = 0
            else:
                scheduler_counter += 1
                if scheduler_counter >= scheduler_patience:
                    # Reduce learning rate
                    self.learning_rate *= scheduler_factor
                    self.optimizer = self._setup_optimizer()
                    opt_state = self.optimizer.init(self.params)
                    scheduler_counter = 0
                    print(f"Reducing learning rate to {self.learning_rate:.6f}")

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
                f'{self.name}: Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.4f} Val Error: {avg_val_loss:.4f}, Learning Rate: {self.learning_rate:.6f}')

        if draw_loss:
            plt.ioff()
            plt.show()

        # Load best parameters
        if best_params is not None:
            self.params = best_params

        return best_val_error

    def get_documentation(self):
        documentation = {"hyperparameters": {
            "learning_rate": self.initial_learning_rate,  # Use initial learning rate
            "n_hidden_size": self.n_hidden_size,
            "n_hidden_layers": self.n_hidden_layers,
            "n_activation_function": self.activation_name,
        }}
        return documentation