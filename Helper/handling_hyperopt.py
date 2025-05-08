# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:12:35 2024

@author: Jonas Kyrion

Beschreibung:   Entählt die nötigen funktionen für die Hyperparameteroptimierung
"""
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

WINDOWSIZE = 1
CUTOFF = 4
NUMBEROFTRIALS = 100
NUMBEROFEPOCHS = 20

# Function to check if the search_space dictionary is grouped
def is_grouped(search_space):
    return all(isinstance(value, dict) for value in search_space.values())

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
    def __init__(self, search_space, model, data_params=None, data=None, n_epochs=20, pruning = True):
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
        if is_grouped(self.search_space):
            # If the search space is grouped, iterate through each group
            for group, values in self.search_space.items():
                for name, value in values.items():
                    if isinstance(value, tuple) and all(isinstance(x, int) for x in value):
                        params[name] = trial.suggest_int(name, value[0], value[1])
                    elif isinstance(value, tuple) and all(isinstance(x, float) for x in value):
                        params[name] = trial.suggest_float(name, value[0], value[1], log=True if 'learning_rate' in name else False)
                    else:
                        params[name] = trial.suggest_categorical(name, value)
        else:
            # If the search space is not grouped, iterate through the parameters directly
            for name, value in self.search_space.items():
                if isinstance(value, tuple) and all(isinstance(x, int) for x in value):
                    params[name] = trial.suggest_int(name, value[0], value[1])
                elif isinstance(value, tuple) and all(isinstance(x, float) for x in value):
                    params[name] = trial.suggest_float(name, value[0], value[1], log=True if 'learning_rate' in name else False)
                else:
                    params[name] = trial.suggest_categorical(name, value)

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

        # Check if the model is a subclass of mb.BaseModel
        if isinstance(self.model, type) and issubclass(self.model, mb.BaseModel):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_size = X_train.shape[1]
            output_size = y_train.T.shape[0] if len(y_train.shape) > 1 else 1  # handle single output case

            # Use the groups from the grouped search_space if available
            if is_grouped(self.search_space):
                model_params = {key: params[key] for key in self.search_space['model_parameters'].keys()}
                hyp_params = {key: params[key] for key in self.search_space['hyperparameters'].keys()}
            else:
                model_params = {key: value for key, value in params.items() if key != 'learning_rate'}
                hyp_params = {key: value for key, value in params.items() if key == 'learning_rate'}

            # Initialize the model with model parameters and move it to the appropriate device
            model = self.model(input_size=input_size, output_size=output_size, **model_params).to(device)
            # Train the model and get the validation error
            val_error = model.train_model(X_train, y_train, X_val, y_val, **hyp_params, n_epochs=self.n_epochs, trial=trial_prun)
        else:
            # Train the model and get the validation error
            val_error, _ = self.model(X_train, y_train, X_val, y_val, **params, n_epochs=self.n_epochs, trial=trial_prun)

        return val_error

"""
# dient als beispiel muss für jede analyse neu geschrieben werden
def objective(trial, n_epochs):
    '''
    Defines the objective function for hyperparameter optimization of an NN using Optuna.
    (Only as example)
    
    Parameters
    ----------
    trial : optuna.trial.Trial
        A trial object that provides methods to suggest hyperparameter values.
    
    Returns
    -------
    float
        The validation error, which is the objective value to be minimized.
    '''
    # Define the search space for the hyperparameters
    n_layers = trial.suggest_int('n_layers', search_space['n_layers'][0], search_space['n_layers'][1])
    n_neurons = trial.suggest_int('n_neurons', search_space['n_neurons'][0], search_space['n_neurons'][1]) # trial.suggest_int('n_neurons', 2, 128)
    lr = trial.suggest_float('learning_rate', search_space['learning_rate'][0], search_space['learning_rate'][1], log=True)
    #activation = trial.suggest_categorical('activation', search_space['activation'])
    activation = 'ReLU'

    # Load the data    
    X_train, X_val, X_test, y_train, y_val, y_test = hda.load_data(data_params)

    # Train the neural network and compute the validation error
    val_error, model = mnn.train_model(X_train, y_train, X_val, y_val, n_layers, n_neurons, activation, lr, NUMBEROFEPOCHS)
    
    # Report intermediate values to the pruner
    trial.report(val_error, n_epochs)

    # Handle pruning based on the intermediate value
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
        
    # Return the validation error as the objective value
    return val_error
"""
def optimize(objective, folderpath, study_name, n_trials=100, n_reduction_factor=3, show_plots=True):
    """
   Optimizes the hyperparameters of a model using Optuna's hyperparameter optimization framework.

   This function creates an Optuna study and optimizes the given objective function.
   It saves the study results, visualizes the optimization progress, and displays
   the best hyperparameters found.

   Parameters
   ----------
   objective : Objective
       The objective function to be minimized, which includes the model training process.
   search_space : dict
       Dictionary defining the search space for hyperparameters, specifying ranges or categories for each parameter.
   folderpath : str
       The directory path where study results and search space details are saved.
   study_name : str
       A unique name for the study, which will be used in naming saved files.
   n_trials : int, optional
       Number of optimization trials to run. Default is 100.
   n_epochs : int, optional
       Number of epochs for training the model within each trial. Default is 20.
   n_reduction_factor : int, optional
       Factor by which to reduce resources in the Hyperband pruning strategy. Default is 3.
   show_plots : bool, optional
       Whether to display optimization and hyperparameter search visualizations. Default is True.
   **kwargs : dict
       Additional keyword arguments that are passed to the objective function.

   Returns
   -------
   optuna.trial.FrozenTrial
       The trial with the best objective value (lowest validation error) found during the optimization.

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

    if(objective.pruning):
        study = optuna.create_study(
            direction='minimize', 
            study_name=study_name, 
            storage=storage_name,
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

    return best_trial

def GetOptimalParameter(folderpath = 'Hyperparameteroptimization', filter_search_space = None, plot = False):
    """
    Retrieves the optimal hyperparameters from saved studies.
 
    Parameters
    ----------
    folderpath : str, optional
        The path to the folder where the study results are saved. The default is 'Hyperparameteroptimization'.
    filter_search_space : dict, optional
        A dictionary defining the filter criteria for the search space. The default is None.
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
            # # Extract the time stamp from the file name
            # parts = filename.split('_')
            # time_parts = parts[-6:]  # Take the last 6 parts as the time stamp
            # time = '_'.join(time_parts).split('.')[0]  # Join them and remove the .db extension

            study_name = os.path.splitext(filename)[0]  # Extract the study name from the file name
            storage_name = "sqlite:///{}/{}".format(folderpath, filename)  # Define the storage name using the file path and name
            study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)  # Load the study from the file
            best_trial = study.best_trial
            
            # # Load the corresponding search space from a file
            # with open('{folderpath}\\search_space_{time}.json'.format(folderpath=folderpath, time=time), 'r') as f:
            #     search_space = json.load(f)            
            
            # # Check if the search space contains the filtered key-value pair
            #if filter_search_space is None or not IsSearchSpaceIsSubset(filter_search_space, best_trial.params):
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
                #fig = optuna.visualization.plot_parallel_coordinate(study)
                #fig.show()
                #Plot the parameter relationship as slice plot in a study.
                optuna.visualization.plot_slice(study, params=list(search_space.keys())).show(renderer="browser")
    
    default_folder = '{folderpath}\\default_search_space.json'.format(folderpath=folderpath)
    if os.path.isfile(default_folder):
        # load the default search space if it exist
        with open(default_folder, 'r') as f:
            default_parameter = json.load(f)
            
                
    optimal_search_space = MergeSearchSpace(optimal_trial.params, default_parameter)
            
    print('Optimal Hyperparameters:')
    for name, value in optimal_search_space.items():
        print(f'{name}: {value}')
    print('Best Objective Value: {}'.format(optimal_trial.value))
    print('Mean Objective Value: {}'.format(np.mean(errors)))
    print('Deviation Objective Value: {}'.format(np.std(errors)))
    return optimal_search_space

def IsSearchSpaceIsSubset(filter_search_space, search_space):
    """
    Checks if the filter_search_space is a subset of the search_space.
    
    Parameters
    ----------
    filter_search_space : dict
        The filter criteria for the search space.
    search_space : dict
        The search space to be checked.
    
    Returns
    -------
    bool
        True if filter_search_space is a subset of search_space, False otherwise.
    """
    is_subset = True
    for key, value in filter_search_space.items():
        if key not in search_space or not all(item in search_space[key] for item in value):
            is_subset = False
            break
    return is_subset 

def MergeSearchSpace(optimal_trial, default_parameter):
    """
    Merges the optimal trial parameters with the default parameters.

    Parameters
    ----------
    optimal_trial : dict
        The optimal trial parameters.
    default_parameter : dict
        The default parameters.

    Returns
    -------
    dict
        The merged search space.
    """
    # Create a copy of optimal_trial to avoid modifying it directly
    combined_trial = optimal_trial.copy()
    
    # Update the combined_trial dictionary with keys and values from default_trial
    # that are not already in combined_trial
    for key, value in default_parameter.items():
        if key not in combined_trial:
            combined_trial[key] = value

    # Now combined_trial contains all entries from optimal_trial and any missing entries from default_trial
    return combined_trial

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
