# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:12:35 2024

@author: Jonas Kyrion

Beschreibung:   Entählt die nötigen funktionen für die Hyperparameteroptimierung
"""
import math

#libarie import
import optuna
import optuna.visualization as optuna_viz
import logging
import sys
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from Helper import handling_data as hda
import Models.model_base as mb
import Models.model_neural_net as mnn
# function import
from datetime import datetime
import plotly

WINDOWSIZE = 1
CUTOFF = 4
NUMBEROFTRIALS = 100
NUMBEROFEPOCHS = 20

class Objective:
    """
    Objective class for defining and executing hyperparameter optimization with Optuna.
    
    This class initializes a search space and model training function, loads data,
    and manages pruning functionality during optimization.
    
    Attributes
    ----------
    search_space : dict
        A dictionary defining the hyperparameter search space.
    model : callable
        The model training function that trains the model using `train_model(X_train, y_train, X_val, y_val, **params)`.
    data_params : dict, optional
        Parameters needed to load the data, if data is not directly passed.
    data : tuple, optional
        A tuple of (X_train, X_val, y_train, y_val) if the data is passed directly.
    n_epochs : int
        Number of epochs for training (default is 20).
    """
    def __init__(self, search_space, model:mb.BaseModel, data_params=None, data=None, n_epochs=20, pruning = True):
        """
        Initializes the Objective class with search space, model function, data, and training epochs.

        Parameters
        ----------
        search_space : dict
            Dictionary defining the hyperparameter ranges for optimization.
        model : callable
            Function that takes in data and hyperparameters to train the model.
        data_params : dict, optional
            Parameters for loading the data, if no direct data is provided.
        data : tuple, optional
            Directly passed data in the format (X_train, X_val, y_train, y_val).
        n_epochs : int, optional
            Number of training epochs, default is 20.
        """
        self.search_space = search_space
        self.model = model
        self.data_params = data_params
        self.data = data
        self.n_epochs = n_epochs
        self.pruning = pruning

    def __call__(self, trial):
        """
        Defines the objective function for Optuna, which will be called for each trial.
        
        Extracts hyperparameters from the search space, loads data if necessary,
        trains the model, and handles pruning.
        
        Parameters
        ----------
        trial : optuna.trial.Trial
            An Optuna trial object used to suggest values for the hyperparameters.
        
        Returns
        -------
        float
            The validation error, which is minimized during optimization.
        """

        # Extract hyperparameters dynamically from the search space
        params = {}
        # If the search space is not grouped, iterate through the parameters directly
        for name, value in self.search_space.items():
            correct_type = (isinstance(value, list) or isinstance(value, tuple))
            if correct_type and len(value) > 2:  # Wenn mehr als 2 Werte im Tuple: kategorisch!
                params[name] = trial.suggest_categorical(name, value)
            elif correct_type and all(isinstance(x, int) for x in value):
                params[name] = trial.suggest_int(name, value[0], value[1])  # Nur für echte Bereiche (z. B. (10, 30))
            elif correct_type and all(isinstance(x, float) for x in value):
                params[name] = trial.suggest_float(name, value[0], value[1],
                                                   log=True if name.startswith('learning_rate') else False)
            else:
                params[name] = trial.suggest_categorical(name, value)  # Fallback für Listen oder Einzelwerte

        # Load or use the data
        if self.data is not None:
            X_train, X_val, y_train, y_val = self.data
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = hda.load_data(self.data_params)

        # Handle pruning
        if self.pruning:
            trial_prun = trial
        else:
            trial_prun = None

        # Check if model is a subclass of mb.BaseModel
        if isinstance(self.model, type) and issubclass(self.model, mb.BaseModel):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_size = X_train.shape[1]
            output_size = y_train.T.shape[0] if len(y_train.shape) > 1 else 1  # handle single output case

            # Initialize the model with model parameters and move it to the appropriate device
            model = self.model(input_size=input_size, output_size=output_size, **params).to(device)
            # Train the model and get the validation error
            val_error = model.train_model(X_train, y_train, X_val, y_val, n_epochs=self.n_epochs, trial=trial_prun)
        else:
            # Train the model and get the validation error
            self.model.reset_hyperparameter(**params)
            val_error = self.model.train_model(X_train, y_train, X_val, y_val, n_epochs=self.n_epochs, trial=trial_prun)

        return val_error


def get_minimum_trials_for_grid_search(search_space):
    """
    Berechnet die minimale Anzahl an Trials die für Grid Search benötigt wird,
    um alle Randbereiche (Min/Max) abzudecken.

    Parameters
    ----------
    search_space : dict
        Dictionary mit Parameternamen als Keys und Bereichen/Listen als Values

    Returns
    -------
    int
        Minimale Anzahl Trials für Grid Search mit Randbereich-Abdeckung
    """
    min_trials = 1

    for name, value in search_space.items():
        if isinstance(value, (list, tuple)) and not isinstance(value, tuple) or \
                (isinstance(value, tuple) and len(value) > 2):
            # Kategorische Parameter - alle Werte müssen getestet werden
            min_trials *= len(value)
        elif isinstance(value, tuple) and len(value) == 2:
            # Parameter mit Bereichen - mindestens Min/Max (2 Werte)
            min_trials *= 2
        else:
            # Einzelwert - 1 Wert
            min_trials *= 1

    return min_trials

def _create_grid_search_space(search_space, n_trials):
    grid_search_space = {}
    range_params = []
    categorical_params = []

    # Trenne kategorische und Bereichsparameter
    for name, value in search_space.items():
        if isinstance(value, list):
            categorical_params.append(name)  # Listen sind immer kategorisch
        elif isinstance(value, tuple):
            # Prüfe, ob es ein numerischer Bereich ist
            if len(value) == 2:
                # Prüfe, ob beide Werte numerisch sind (int oder float)
                is_numeric = all(isinstance(x, (int, float)) for x in value)
                if is_numeric:
                    range_params.append(name)  # Numerischer Bereich
                else:
                    categorical_params.append(name)  # Nicht-numerisch → kategorisch
            else:
                categorical_params.append(name)  # Tuples mit ≠2 Werten → kategorisch
        else:
            categorical_params.append(name)  # Einzelwerte → kategorisch

    # Berechne kategorische Kombinationen
    categorical_combinations = 1
    for name in categorical_params:
        value = search_space[name]
        if isinstance(value, (list, tuple)):
            categorical_combinations *= len(value)

    available_trials = n_trials / categorical_combinations if categorical_combinations > 0 else n_trials
    n_range_params = len(range_params)

    # Berechne optimale k (Werte pro Bereichsparameter)
    if n_range_params > 0:
        k = int(round((available_trials)**(1/n_range_params), 6))
    else:
        k = 2

    # Erstelle Grid-Search-Werte
    for name, value in search_space.items():
        if isinstance(value, (list, tuple)) and (len(value) > 2 or not isinstance(value, tuple)):
            grid_search_space[name] = list(value)  # Kategorisch: alle Werte
        elif isinstance(value, tuple) and len(value) == 2:
            min_val, max_val = value
            if name.startswith('n_') or all(isinstance(x, int) for x in value):
                # Integer: Gleichmäßige Verteilung
                if k == 2:
                    grid_search_space[name] = [min_val, max_val]
                else:
                    step = int((max_val - min_val) / (k - 1))
                    grid_search_space[name] = [min_val + i * step for i in range(k)]
            elif all(isinstance(x, float) for x in value):
                # Float: Lineare oder logarithmische Verteilung
                if name.startswith(('learning_rate', 'log_', 'lr')):
                    log_min, log_max = math.log10(min_val), math.log10(max_val)
                    log_values = np.linspace(log_min, log_max, k)
                    grid_search_space[name] = [10**x for x in log_values]
                else:
                    grid_search_space[name] = list(np.linspace(min_val, max_val, k))
            else:
                grid_search_space[name] = list(value)
        else:
            grid_search_space[name] = [value]

    return grid_search_space

def optimize(objective, folderpath, study_name, n_trials=100, n_reduction_factor=3, sampler="TPESampler",
             show_plots=True):
    """
   Optimizes the hyperparameters of a model using Optuna's hyperparameter optimization framework.

   This function creates an Optuna study and optimizes the given objective function.
   It saves the study results, visualizes the optimization progress, and displays
   the best hyperparameters found.

   Parameters
   ----------
   objective : Objective
       The objective function to be minimized, which includes the model training process.
   folderpath : str
       The directory path where study results and search space details are saved.
   study_name : str
       A unique name for the study, which will be used in naming saved files.
   n_trials : int, optional
       Number of optimization trials to run. Default is 100.
   n_reduction_factor : int, optional
       Factor by which to reduce resources in the Hyperband pruning strategy. Default is 3.
   sampler : str, optional
       Sampler used for optimization. Possible Sampler are 'RandomSampler', 'GridSampler', 'TPESampler'. Default is 'TPESampler'.
       For more information see https://optuna.readthedocs.io/en/stable/reference/samplers/index.html#module-optuna.samplers
   show_plots : bool, optional
       Whether to display optimization and hyperparameter search visualizations. Default is True.

   Returns
   -------
   dict
       The best hyperparameters found during the optimization.

   Saves
   -----
   Saves the search space and study results in `folderpath` for later retrieval.

   Notes
   -----
   This function uses the `HyperbandPruner` for efficient pruning, and it can visualize the
   optimization history, parallel coordinates of parameters, and other aspects of the search space
   if `show_plots` is set to True.
   """
    now = datetime.now()
    time = now.strftime("%Y_%m_%d_%H_%M_%S")
    study_name = study_name + time
    storage_name = f"sqlite:///{folderpath}\\{study_name}.db"

    # Erstelle den entsprechenden Sampler
    if sampler == "TPESampler":
        sampler_opt = optuna.samplers.TPESampler()
    elif sampler == "RandomSampler":
        sampler_opt = optuna.samplers.RandomSampler()
    elif sampler == "GridSampler":
        # Konvertiere den Suchraum für Grid Search
        grid_search_space = _create_grid_search_space(objective.search_space, n_trials)
        sampler_opt = optuna.samplers.GridSampler(grid_search_space)

        # Berechne die tatsächliche Anzahl der Trials für Grid Search
        actual_trials = 1
        for param_values in grid_search_space.values():
            actual_trials *= len(param_values)

        print(f"Grid Search konfiguriert:")
        print(f"  Gewünschte Trials: {n_trials}")
        print(f"  Tatsächliche Trials: {actual_trials}")
        print(f"  Grid Search Space:")
        for name, values in grid_search_space.items():
            print(f"    {name}: {len(values)} Werte -> {values[:3]}{'...' if len(values) > 3 else ''}")

        # Aktualisiere n_trials für Grid Search
        n_trials = actual_trials
    else:
        print("No valid sampler was selected. TPESampler will be used.")
        sampler_opt = optuna.samplers.TPESampler()

    # Erstelle die Studie
    if objective.pruning and sampler != "GridSampler":
        # Pruning funktioniert nicht gut mit Grid Search
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage_name,
            sampler=sampler_opt,
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=n_trials,
                reduction_factor=n_reduction_factor
            )
        )
    else:
        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage_name,
            sampler=sampler_opt
        )

    study.optimize(objective, n_trials=n_trials)

    # Zeige Ergebnisse und Plots an
    best_trial = study.best_trial
    print("Best Hyperparameters:", best_trial.params)
    print("Best Objective Value:", best_trial.value)

    if show_plots:
        optuna_viz.plot_optimization_history(study).show()
        optuna_viz.plot_parallel_coordinate(study, params=list(objective.search_space.keys())).show()
        optuna_viz.plot_slice(study, params=list(objective.search_space.keys())).show()
        optuna_viz.plot_param_importances(study).show()

    return best_trial.params

def GetOptimalParameter(folderpath = 'Hyperparameteroptimization', plot = False):
    """
    Retrieves the optimal hyperparameters from saved studies.
 
    Parameters
    ----------
    folderpath : str, optional
        The path to the folder where the study results are saved. The default is 'Hyperparameteroptimization'.
    plot : bool, optional
        Whether to plot the optimization results. The default is False.
 
    Returns
    -------
    dict
        The optimal hyperparameters.
    """
    # Define a variable to store the best trial
    optimal_trial = None
    
    # define a variable to store the result 
    errors = []
    
    # Iterate over every file in the path, load and visualize the study
    for filename in os.listdir(folderpath):
        if filename.endswith('.db'):  # Only load files with the .db extension
            # # Extract the time stamp from the file name
            # parts = filename.split('_')
            # time_parts = parts[-6:]  # Take the last 6 parts as the time stamp
            # time = '_'.join(time_parts).split('.')[0]  # Join them and remove the .db extension

            study_name = os.path.splitext(filename)[0]  # Extract the study name from the file name
            storage_name = "sqlite:///{}/{}".format(folderpath, filename)  # Define the storage name using the file path and name
            study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)  # Load the study from the file
            best_trial = study.best_trial

            # # Check if the search space contains the filtered key-value pair
            # otherwise Skip this search space and move to the next one
            
            # Print the best hyperparameters and the best objective value                
            print('Best Hyperparameters:')
            for name, value in best_trial.params.items():
                print(f'{name}: {value}')
            print('Best Objective Value: {}'.format(best_trial.value))
            errors.append(best_trial.value)
            
            # Update the variable to store the best trial
            if optimal_trial is None or best_trial.value < optimal_trial.value:
                optimal_trial = best_trial
            
            if plot:
                #plot
                plt.figure(dpi=1200)
                # Plot the hyperparameter search space
                fig = optuna.visualization.plot_parallel_coordinate(study)
                fig.show()

    optimal_search_space = optimal_trial.params
            
    print('Optimal Hyperparameters:')
    for name, value in optimal_search_space.items():
        print(f'{name}: {value}')
    print('Best Objective Value: {}'.format(optimal_trial.value))
    print('Mean Objective Value: {}'.format(np.mean(errors)))
    print('Deviation Objective Value: {}'.format(np.std(errors)))
    return optimal_search_space

# Schreibt den übergeben search_space in eine default datei
def WriteAsDefault(folderpath, search_space):
    """
    Writes the search space to a default file.
    
    Parameters
    ----------
    folderpath : str
        The path to the folder where the default file will be saved.
    search_space : dict
        The search space to be saved.
    
    Returns
    -------
    None
    """
    # save the search_space
    with open('{folderpath}\\default_search_space.json'.format(folderpath=folderpath), 'w') as f:
        json.dump(search_space, f)

def load_search_spaces(json_path: str) -> dict[str, any]:
    """
    Loads the search spaces for machine learning models from a JSON file.

    This function reads a JSON file containing search spaces for different models (e.g., 'RF' for Random Forest,
    'NN' for Neural Network). It converts lists with exactly two elements into tuples, which is useful for
    hyperparameter optimization libraries like Optuna.

    Args:
        json_path (str): Path to the JSON file containing the search spaces.

    Returns:
        Dict[str, Any]: A dictionary where keys are model names (e.g., 'RF', 'NN') and values are dictionaries
                        of search spaces. Lists with two elements are converted to tuples.

    Example:
        >>> search_spaces = load_search_spaces("search_spaces.json")
        >>> print(search_spaces["RF"])
        {
            'n_estimators': (100, 500),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 10),
            'max_depth': (100, 500)
        }
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    search_spaces = data.get("search_spaces", {})

    for model_name, space in search_spaces.items():
        for key, value in space.items():
            if isinstance(value, list) and len(value) == 2:
                search_spaces[model_name][key] = tuple(value)

    return search_spaces
