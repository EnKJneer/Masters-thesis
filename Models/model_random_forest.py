import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import Models.model_base as mb

class RandomForestModel(mb.BaseModel):
    def __init__(self, n_estimators=100, max_features = None, max_depth =None, min_samples_split = 2, min_samples_leaf = 1, random_state=42, name ="Random_Forest"):
        """
        Initializes a Random Forest regressor.

        Parameters
        ----------
        n_estimators : int, optional
            The number of trees in the Random Forest. The default is 100.
        random_state : int, optional
            Controls the randomness of the estimator. The default is None.
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_features = max_features, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, random_state=random_state, n_jobs = -1)
        self.scaler = None

        # Save Parameter for documentation
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.name = name
        self.device = "cpu"

    def reset_hyperparameter(self, n_estimators=100, max_features = None, max_depth =None, min_samples_split = 2, min_samples_leaf = 1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        # Neuinitialisierung des Modells mit den neuen Hyperparametern
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
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
        Make predictions based on the input data using the Random Forest model.

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

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=1, trial=None, draw_loss=False, n_outlier=12, patience=10):
        """
        Train the Random Forest model using the training data and validate it using the validation data.

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
            The number of epochs for training. Default is 1 (since Random Forest is not iterative).
            Only for compatibility reasons.
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
        # Training loop (for compatibility, though Random Forest is not iterative)

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

        print(
            f'{self.name}: Val Error: {val_error:.4f}')

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
        documentation = {"hyperparameters": {
            "n_estimators": self.n_estimators,
            "max_features": self.max_features,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf
        }}
        return documentation
    @staticmethod
    def get_reference_model():
        return RandomForestModel(n_estimators=10, max_features = None, min_samples_split = 2, min_samples_leaf = 1)

class ExtraTreesModel(mb.BaseModel):
    def __init__(self, n_estimators=100, max_features=1, min_samples_split=2, min_samples_leaf=1, random_state=42, name="Extra_Trees"):
        """
        Initializes an Extra Trees regressor.

        Parameters
        ----------
        n_estimators : int, optional
            The number of trees in the Extra Trees. The default is 100.
        random_state : int, optional
            Controls the randomness of the estimator. The default is None.
        """
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = None

        # Save Parameter for documentation
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.name = name
        self.device = "cpu"

    def reset_hyperparameter(self, n_estimators=100, max_features = None, max_depth =None, min_samples_split = 2, min_samples_leaf = 1):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        # Neuinitialisierung des Modells mit den neuen Hyperparametern
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
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
        return mean_squared_error(y_target.squeeze(), y_pred.squeeze())

    def predict(self, X):
        """
        Make predictions based on the input data using the Extra Trees model.

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

    def train_model(self, X_train, y_train, X_val, y_val, n_epochs=1, trial=None, draw_loss=False, n_outlier=12, patience=10):
        """
        Train the Extra Trees model using the training data and validate it using the validation data.

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
            The number of epochs for training. Default is 1 (since Extra Trees is not iterative).
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

        # Training loop (for compatibility, though Extra Trees is not iterative)
        for epoch in range(n_epochs):
            self.model.fit(X_train, y_train.squeeze())
            y_val_pred = self.model.predict(X_val)
            val_error = self.criterion(y_val, y_val_pred)

            # Report intermediate values to the pruner
            if trial:
                trial.report(val_error, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Update the best validation error
            if val_error < best_val_error:
                best_val_error = val_error

            if draw_loss:
                plt.plot(epoch, val_error, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

            print(f'{self.name}: Epoch {epoch + 1}/{n_epochs}, Val Error: {val_error:.4f}')

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
        documentation = {
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_features": self.max_features,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf
            }
        }
        return documentation
    @staticmethod
    def get_reference_model():
        return ExtraTreesModel(n_estimators=10, max_features = None, min_samples_split = 2, min_samples_leaf = 1)