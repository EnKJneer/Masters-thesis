import os
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf

""" Functions """
def hyperparameter_optimization_quantilNet(folder_path, X_train, X_val, y_train, y_val):
    # Hyperparameter optimization for neural network
    study_name_nn = "Hyperparameter_Neural_Net_QId"

    default_parameter_nn = {
        'activation': 'ReLU',
        'window_size': window_size,
        'past_values': past_values,
        'future_values': future_values,
    }
    num_db_files_nn = sum(file.endswith('.db') for file in os.listdir(folder_path))

    while num_db_files_nn < 5:
        search_space_nn = {
            'learning_rate': (0.5e-3, 8e-2),
            'n_neurons': (15, 128),
            'n_layers': (3, 12),
            'output_distribution': ('uniform', 'normal')
        }
        # Erstelle das Objective
        objective_nn = hyperopt.Objective(
            search_space=search_space_nn,
            model=mnn.QuantileIdNetModel,
            data=[X_train, X_val, y_train, y_val],
            n_epochs=NUMBEROFEPOCHS,
            pruning=True
        )
        hyperopt.optimize(objective_nn, folder_path, study_name_nn, n_trials=NUMBEROFTRIALS)
        hyperopt.WriteAsDefault(folder_path, default_parameter_nn)
        num_db_files_nn = sum(file.endswith('.db') for file in os.listdir(folder_path))

    # Load hyperparameters for neural network
    model_params = hyperopt.GetOptimalParameter(folder_path, plot=False)
    print("Neural Network Hyperparameters:", model_params)
    return model_params

def quantilNet_prediction(data_params, model_params):

    y_predicted = pd.DataFrame()
    # Create and train the neural network and compute the validation error, for each axis separately
    input_size = X_train.shape[1]

    axis = data_params.target_channels[0]

    output_size = y_train[axis].T.shape[0] if len(y_train[axis].shape) > 1 else 1  # handle single output case

    # Initialize a list to store predictions from all models
    all_predictions = []

    for i in range(0, NUMBEROFMODELS):
        # Initialize the model
        model_nn = mnn.QuantileIdNetModel(input_size=input_size, output_size=output_size,
                           n_neurons=model_params['n_neurons'], n_layers=model_params['n_layers'],
                           activation=nn.ReLU)

        # Train the model
        val_error = model_nn.train_model(
            X_train, y_train[axis], X_val, y_val[axis],
            learning_rate=model_params['learning_rate'], n_epochs=NUMBEROFEPOCHS, patience=5
        )
        loss, predictions = model_nn.test_model(X_test, y_test[axis])

        # Store the predictions
        all_predictions.append(predictions.flatten())

    # Calculate the mean of all predictions
    mean_predictions = np.mean(all_predictions, axis=0)

    hplot.plot_prediction_vs_true('Neural_Net_qauantilID ' + data_params.name + ' ' + axis, mean_predictions.T, y_test[axis])

    # Store the mean predictions
    y_predicted[axis + "_pred"] = mean_predictions

    return y_predicted

def hyperparameter_optimization_ml(folder_path):
    # Hyperparameter optimization for neural network
    study_name_nn = "Hyperparameter_Neural_Net_"

    default_parameter_nn = {
        'activation': 'ReLU',
        'window_size': window_size,
        'past_values': past_values,
        'future_values': future_values,
    }
    num_db_files_nn = sum(file.endswith('.db') for file in os.listdir(folder_path))

    while num_db_files_nn < 5:
        search_space_nn = {
            'learning_rate': (0.5e-3, 8e-2),
            'n_neurons': (15, 128),
            'n_layers': (3, 12),
        }
        # Erstelle das Objective
        objective_nn = hyperopt.Objective(
            search_space=search_space_nn,
            model=mnn.Net,
            data=[X_train, X_val, y_train["curr_x"], y_val["curr_x"]],
            n_epochs=NUMBEROFEPOCHS,
            pruning=True
        )
        hyperopt.optimize(objective_nn, folder_path, study_name_nn, n_trials=NUMBEROFTRIALS)
        hyperopt.WriteAsDefault(folder_path, default_parameter_nn)
        num_db_files_nn = sum(file.endswith('.db') for file in os.listdir(folder_path))

    # Load hyperparameters for neural network
    model_params = hyperopt.GetOptimalParameter(folder_path, plot=False)
    print("Neural Network Hyperparameters:", model_params)

    return model_params

def nn_prediction(data_params, model_params):

    y_predicted = pd.DataFrame()
    # Create and train the neural network and compute the validation error, for each axis separately
    input_size = X_train.shape[1]

    axis = data_params.target_channels[0]

    output_size = y_train[axis].T.shape[0] if len(y_train[axis].shape) > 1 else 1  # handle single output case

    # Initialize a list to store predictions from all models
    all_predictions = []

    for i in range(0, NUMBEROFMODELS):
        # Initialize the model
        model_nn = mnn.Net(input_size=input_size, output_size=output_size,
                           n_neurons=model_params['n_neurons'], n_layers=model_params['n_layers'],
                           activation=nn.ReLU)

        # Train the model
        val_error = model_nn.train_model(
            X_train, y_train[axis], X_val, y_val[axis],
            learning_rate=model_params['learning_rate'], n_epochs=NUMBEROFEPOCHS, patience=5
        )
        loss, predictions = model_nn.test_model(X_test, y_test[axis])

        # Store the predictions
        all_predictions.append(predictions.flatten())

    # Calculate the mean of all predictions
    mean_predictions = np.mean(all_predictions, axis=0)

    hplot.plot_prediction_vs_true('Neural_Net ' + data_params.name + ' ' + axis, mean_predictions.T, y_test[axis])

    # Store the mean predictions
    y_predicted[axis + "_pred"] = mean_predictions

    return y_predicted

def hyperparameter_optimization_riemann(folder_path, X_train, X_val, y_train, y_val):
    study_name_nn = "Hyperparameter_Neural_Net_Riemann"

    default_parameter_nn = {
        'activation': 'ReLU',
        'window_size': window_size,
        'past_values': past_values,
        'future_values': future_values,
    }
    num_db_files_nn = sum(file.endswith('.db') for file in os.listdir(folder_path))

    while num_db_files_nn < 5:
        search_space_nn = {
            'learning_rate': (0.5e-3, 8e-2),
            'n_neurons': (15, 128),
            'n_layers': (3, 12),
        }
        objective_nn = hyperopt.Objective(
            search_space=search_space_nn,
            model=mnn.RiemannQuantileClassifierNet,
            data=[X_train, X_val, y_train, y_val],
            n_epochs=NUMBEROFEPOCHS,
            pruning=True
        )
        hyperopt.optimize(objective_nn, folder_path, study_name_nn, n_trials=NUMBEROFTRIALS)
        hyperopt.WriteAsDefault(folder_path, default_parameter_nn)
        num_db_files_nn = sum(file.endswith('.db') for file in os.listdir(folder_path))

    model_params = hyperopt.GetOptimalParameter(folder_path, plot=False)
    print("Riemann Quantile Classifier Hyperparameters:", model_params)
    return model_params

def riemannNet_prediction(data_params, model_params):
    y_predicted = pd.DataFrame()
    input_size = X_train.shape[1]

    axis = data_params.target_channels[0]
    output_size = 1  # Klassifikation

    all_predictions = []

    for i in range(0, NUMBEROFMODELS):
        model_nn = mnn.RiemannQuantileClassifierNet(
            input_size=input_size,
            output_size=output_size,
            n_neurons=model_params['n_neurons'],
            n_layers=model_params['n_layers'],
            activation=nn.ReLU
        )

        val_error = model_nn.train_model(
            X_train, y_train[axis], X_val, y_val[axis],
            learning_rate=model_params['learning_rate'], n_epochs=NUMBEROFEPOCHS, patience=5
        )
        loss, predictions = model_nn.test_model(X_test, y_test[axis])
        all_predictions.append(predictions.flatten())

    mean_predictions = np.mean(all_predictions, axis=0)
    hplot.plot_prediction_vs_true('Neural_Net_Riemann ' + data_params.name + ' ' + axis, mean_predictions.T, y_test[axis])
    y_predicted[axis + "_pred"] = mean_predictions
    return y_predicted

def hyperparameter_optimization_rf(folder_path):
    # Hyperparameter optimization for random forest
    study_name_rf = "Hyperparameter_RF_mini_"

    default_parameter_rf = {
        'window_size': window_size,
        'past_values': past_values,
        'future_values': future_values,
    }
    num_db_files_rf = sum(file.endswith('.db') for file in os.listdir(folder_path))

    while num_db_files_rf < 4:
        search_space_rf = {
            'n_estimators': (5, 50),
            'max_features': ['sqrt', 'log2', None],
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 4)
        }
        # Erstelle das Objective
        objective_rf = hyperopt.Objective(
            search_space=search_space_rf,
            model=mrf.RandomForestModel,
            data=[X_train, X_val, y_train["curr_x"], y_val["curr_x"]],
            n_epochs=NUMBEROFEPOCHS,
            pruning=True
        )
        hyperopt.optimize(objective_rf, folder_path_rf, study_name_rf, n_trials=NUMBEROFTRIALS)
        hyperopt.WriteAsDefault(folder_path_rf, default_parameter_rf)
        num_db_files_rf = sum(file.endswith('.db') for file in os.listdir(folder_path_rf))

    # Load hyperparameters for random forest
    model_params = hyperopt.GetOptimalParameter(folder_path, plot=False)
    print("Random Forest Hyperparameters:", model_params)

    return model_params

def rf_prediction(data_params, model_params, name='Random_forest_mini_en'):
    y_predicted = pd.DataFrame()


    # Create and train the random forest and compute the validation error, for each axis separately
    for axis in data_params.target_channels:
        # Initialize a list to store predictions from all models
        all_predictions = []

        # Initialize the model
        model_rf = mrf.RandomForestModel(n_estimators=model_params['n_estimators'],
                                         max_features=model_params['max_features'],
                                         min_samples_leaf=model_params['min_samples_leaf'],
                                         min_samples_split=model_params['min_samples_split'])

        # Train the model
        val_error = model_rf.train_model(X_train, y_train[axis], X_val, y_val[axis])
        loss, predictions = model_rf.test_model(X_test, y_test[axis])

        hplot.plot_prediction_vs_true(name + ' ' + data_params.name + ' ' + axis, predictions.T, y_test[axis])

        # Store the mean predictions
        y_predicted[axis + "_pred"] = predictions

    return y_predicted

""" Data Sets """
folder_data = '..\\DataSets\\DataFiltered'
dataSet_same_material_diff_workpiece = hdata.DataclassCombinedTrainVal('Al_Al_Gear_Plate', folder_data,
                                                                       ['AL_2007_T4_Gear_Depth_3.csv','AL_2007_T4_Gear_Normal_1.csv'],
                                                                       ['AL_2007_T4_Gear_SF_2.csv', 'AL_2007_T4_Gear_Normal_2.csv'],
                                                                       ['AL_2007_T4_Plate_Normal_3.csv'],
                                                                       ["curr_x"])
dataSet_diff_material_same_workpiece = hdata.DataclassCombinedTrainVal('Al_St_Gear_Gear', folder_data,
                                                                       ['AL_2007_T4_Gear_Depth_3.csv','AL_2007_T4_Gear_Normal_1.csv'],
                                                                       ['AL_2007_T4_Gear_SF_2.csv', 'AL_2007_T4_Gear_Normal_2.csv'],
                                                                       ['S235JR_Gear_Normal_3.csv'],
                                                                       ["curr_x"])
dataSet_diff_material_diff_workpiece = hdata.DataclassCombinedTrainVal('Al_St_Gear_Plate', folder_data,
                                                                       ['AL_2007_T4_Gear_Depth_3.csv', 'AL_2007_T4_Gear_Normal_1.csv'],
                                                                       ['AL_2007_T4_Gear_SF_2.csv', 'AL_2007_T4_Gear_Normal_2.csv'],
                                                                       ['S235JR_Plate_Normal_3.csv'],
                                                                       ["curr_x"])

dataSets_list = [dataSet_same_material_diff_workpiece,
                 dataSet_diff_material_same_workpiece,
                 dataSet_diff_material_diff_workpiece]


if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 10

    window_size = 10
    past_values = 2
    future_values = 2

    torch.cuda.is_available()

    X_train, X_val, X_test, y_train, y_val, y_test = hdata.load_data(dataSets_list[0], past_values=past_values,
                                                                     future_values=future_values,
                                                                     window_size=window_size)
    """ Hyperparameter """
    folder_path_nn = '../Models/Hyperparameter/NeuralNet_curr_x'
    model_params_nn = hyperparameter_optimization_ml(folder_path_nn)

    folder_path_rf = '../Models/Hyperparameter/RandomForest_mini'
    model_params_rf = hyperparameter_optimization_rf(folder_path_rf)

    folder_path_nn = '../Models/Hyperparameter/QuantilIDNet'
    model_params_quantilNet = hyperparameter_optimization_quantilNet(folder_path_nn, X_train, X_val, y_train, y_val)

    folder_path_rf = '../Models/Hyperparameter/RandomForest_mini_moreData'
    model_params_rf_2 = hyperparameter_optimization_rf(folder_path_rf)

    folder_path_nn = '../Models/Hyperparameter/RiemannQuantileNet'
    model_params_riemann = hyperparameter_optimization_riemann(folder_path_nn, X_train, X_val, y_train, y_val)

    """ Prediction """
    # Save Meta information
    # Define the meta information structure
    meta_information = {
        "DataSets": [],
        "Models": {
            "Neural_Net_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                "hyperparameters": {
                    "learning_rate": model_params_nn['learning_rate'],
                    "n_neurons": model_params_nn['n_neurons'],
                    "n_layers": model_params_nn['n_layers']
                }
            },
            "Quantil_Net_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                "hyperparameters": {
                    "learning_rate": model_params_quantilNet['learning_rate'],
                    "n_neurons": model_params_quantilNet['n_neurons'],
                    "n_layers": model_params_quantilNet['n_layers'],
                    "output_distribution": model_params_quantilNet['output_distribution']
                }
            },
            "Riemann_Quantile_Net_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                "hyperparameters": {
                    "learning_rate": model_params_riemann['learning_rate'],
                    "n_neurons": model_params_riemann['n_neurons'],
                    "n_layers": model_params_riemann['n_layers']
                }
            },
            "Random_Forest_mini_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                "hyperparameters": {
                    "n_estimators": model_params_rf['n_estimators'],
                    "max_features": model_params_rf['max_features'],
                    "min_samples_split": model_params_rf['min_samples_split'],
                    "min_samples_leaf": model_params_rf['min_samples_leaf']
                }
            },
            "Random_Forest": {
                "hyperparameters": {
                    "n_estimators": model_params_rf['n_estimators']*NUMBEROFMODELS,
                    "max_features": model_params_rf['max_features'],
                    "min_samples_split": model_params_rf['min_samples_split'],
                    "min_samples_leaf": model_params_rf['min_samples_leaf']
                }
            },
            "Random_Forest_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                "hyperparameters": {
                    "n_estimators": model_params_rf_2['n_estimators'],
                    "max_features": model_params_rf_2['max_features'],
                    "min_samples_split": model_params_rf_2['min_samples_split'],
                    "min_samples_leaf": model_params_rf_2['min_samples_leaf']
                }
            }
        },
        "Data_Preprocessing": {
            "window_size": window_size,
            "past_values": past_values,
            "future_values": future_values
        }
    }

    # Populate the DataSets section
    for data_params in dataSets_list:
        data_info = {
            "name": data_params.name,
            "folder": data_params.folder,
            "training_data_paths": data_params.training_data_paths,
            "validation_data_paths": data_params.validation_data_paths,
            "testing_data_paths": data_params.testing_data_paths,
            "target_channels": data_params.target_channels,
            "percentage_used": data_params.percentage_used
        }
        meta_information["DataSets"].append(data_info)

    # Save the meta information to a JSON file
    with open('Data/info.json', 'w') as json_file:
        json.dump(meta_information, json_file, indent=4)

    n_estimators = model_params_rf['n_estimators']
    # Predict
    for data_params in dataSets_list:
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = hdata.load_data(data_params, past_values=past_values, future_values=future_values, window_size=window_size)
        training_data, val_data, test_data, file_name = hdata.load_filtered_data(data_params, past_values=past_values, future_values=future_values, window_size=window_size)

        # Save data for visualization
        data = test_data[0]

        # Neural Net
        # Assign predictions to test_data
        data["curr_x_pred_NN_en"] = nn_prediction(data_params, model_params_nn)
        # save data
        file_path = f'Data/{data_params.name}.csv'
        hdata.save_data(data, file_path)

        # Random Forest
        data = pd.DataFrame()
        header = ["curr_x_pred_RF_mini_en", "curr_x_pred_RF_en", "curr_x_pred_RF", "curr_x_pred_NN_quanti_en", "curr_x_pred_NN_riemann_en"]
        # Assign predictions to test_data
        model_params_rf['n_estimators'] = n_estimators
        data["curr_x_pred_RF_mini_en"] = rf_prediction(data_params, model_params_rf)
        data["curr_x_pred_RF_en"] = rf_prediction(data_params, model_params_rf_2, name ='Random_forest_en')
        model_params_rf['n_estimators'] = n_estimators*NUMBEROFMODELS
        data["curr_x_pred_RF"] = rf_prediction(data_params, model_params_rf_2, name='Random_forest')
        data["curr_x_pred_NN_quanti_en"] = quantilNet_prediction(data_params, model_params_quantilNet)
        data["curr_x_pred_NN_riemann_en"] = riemannNet_prediction(data_params, model_params_riemann)

        # Add data to file
        # Check if file exists
        hdata.add_pd_to_csv(file_path, data, header)