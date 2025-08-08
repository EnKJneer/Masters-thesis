import copy
import datetime
import json
import os
import numpy as np
import pandas as pd
import torch
from numpy.f2py.auxfuncs import throw_error
from scipy.optimize import curve_fit
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
#import Models.model_mixture_of_experts as mmix
from datetime import datetime

def run_experiment(dataSets, models, search_spaces, optimization_samplers=["TPESampler", "RandomSampler", "GridSampler"],
                   NUMBEROFEPOCHS=800, NUMBEROFMODELS=10, NUMBEROFTRIALS = 10,
                   patience=5, plot_types=None, prunning = True,
                   experiment_name = 'Experiment'):

    if type(dataSets) is not list:
        dataSets = [dataSets]
    if type(models) is not list:
        models = [models]

    # Create directory for results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = os.path.join(experiment_name, timestamp)
    results_dir = os.path.join('Results', results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Define the meta information structure
    meta_information = {
        "DataSets": [],
        "Models": [],
        "Data_Preprocessing": {
            "Epochs": NUMBEROFEPOCHS,
            "Trials": NUMBEROFTRIALS,
            "Pruning": prunning,
        }
    }

    # Define reference models
    reference_models = []
    for model in models:
        reference_models.append(model.get_reference_model())

    # Add datasets to meta information
    for data_params in dataSets:
        data_info = {
            **data_params.get_documentation()
        }
        meta_information["DataSets"].append(data_info)

    """ Prediction """
    results = []
    reference_models_copy = copy.deepcopy(reference_models)
    for i, dataClass in enumerate(dataSets):
        print(f"\n===== Verarbeitung: {dataClass.name} =====")
        # Daten laden
        X_train, X_val, X_test, y_train, y_val, y_test = dataClass.load_data()
        raw_data = dataClass.load_raw_test_data()
        if isinstance(X_test, list):
            df_list_results = [pd.DataFrame() for x in range(len(X_test))]
            header_list = [[] for x in range(len(X_test))]
        else:
            df_list_results = [pd.DataFrame()]
            header_list = []

        """ Hyperparameteroptimization"""
        models_optimized = []
        for idx, model in enumerate(models):
            for sampler in optimization_samplers:
                study_name = experiment_name + '_' + sampler +'_'
                search_space = search_spaces[idx]
                objective_nn = hyperopt.Objective(
                    search_space=search_space,
                    model=copy.copy(model),
                    data=[X_train, X_val, y_train, y_val],
                    n_epochs=NUMBEROFEPOCHS,
                    pruning=prunning
                )
                best_params = hyperopt.optimize(objective_nn, results_dir, study_name=study_name, n_trials=NUMBEROFTRIALS, sampler=sampler)
                model_optimized = copy.deepcopy(model)
                model_optimized.reset_hyperparameter(**best_params)
                model_optimized.name = model_optimized.name + '_' + sampler
                models_optimized.append(model_optimized)

        # Train and test reference models
        for idx, model in enumerate(reference_models):
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            for _ in range(NUMBEROFMODELS):
                model = reference_models_copy[idx]
                model.target_channel = dataClass.target_channels[0]
                model.train_model(X_train, y_train, X_val, y_val, n_epochs = NUMBEROFEPOCHS, patience_stop=patience)
                if isinstance(X_test, list):
                    for i, (x, y) in enumerate(zip(X_test, y_test)):
                        mse, pred_nn = model.test_model(x, y)
                        nn_preds[i].append(pred_nn.flatten())
                        print(f"{model.name}: Test MAE: {mse}")
                else:
                    mse, pred_nn = model.test_model(X_test, y_test)
                    nn_preds.append(pred_nn.flatten())
                    print(f"{model.name}: Test MAE: {mse}")
                reference_models[idx] = model

            # Fehlerberechnung
            hexp.calculate_and_store_results(model, dataClass, nn_preds, y_test, df_list_results, results, header_list,
                                        raw_data)

        # Train and test models
        for idx, model_op in enumerate(models_optimized):
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            for _ in range(NUMBEROFMODELS):
                model = copy.deepcopy(model_op)
                if hasattr(model, 'input_size'):
                    model.input_size = None
                model.target_channel = dataClass.target_channels[0]
                model.train_model(X_train, y_train, X_val, y_val, n_epochs=NUMBEROFEPOCHS, patience_stop=patience)
                if hasattr(model, 'clear_active_experts_log'):
                    model.clear_active_experts_log()  # Clear the log for the next test
                if isinstance(X_test, list):
                    for i, (x, y) in enumerate(zip(X_test, y_test)):
                        mse, pred_nn = model.test_model(x, y)
                        print(f"{model.name}: Test RMAE: {mse}")
                        nn_preds[i].append(pred_nn.flatten())

                        # Check if the model has the method to plot active experts and call it
                        if hasattr(model, 'plot_active_experts'):
                            model.plot_active_experts()
                            model.clear_active_experts_log()  # Clear the log for the next test
                else:
                    mse, pred_nn = model.test_model(X_test, y_test)
                    print(f"{model.name}: Test RMAE: {mse}")
                    nn_preds.append(pred_nn.flatten())

                    # Check if the model has the method to plot active experts and call it
                    if hasattr(model, 'plot_active_experts'):
                        model.plot_active_experts()
                        model.clear_active_experts_log()  # Clear the log for the next test

            # Fehlerberechnung
            hexp.calculate_and_store_results(model, dataClass, nn_preds, y_test, df_list_results, results, header_list,
                                        raw_data)

    # After training to include learned parameters
    # Add reference models to meta information
    for model in reference_models:
        model_info = {
            model.name: {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                **model.get_documentation()
            },
        }
        meta_information["Models"].append(model_info)
    # Add models to meta information
    for model in models_optimized:
        model_info = {
            model.name: {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                **model.get_documentation()
            },
        }
        meta_information["Models"].append(model_info)

    # Save the meta information to a JSON file
    documentation = meta_information

    # ========== MODULARE PLOT-ERSTELLUNG ==========

    # DataFrame mit korrigierter Struktur erstellen
    df = pd.DataFrame(results, columns=hexp.HEADER)

    # Modulare Plots erstellen
    print("\n===== Erstelle Plots =====")
    plot_paths = hexp.create_plots_modular(results_dir, results, plot_types)

    # Plot-Pfade zur Dokumentation hinzufügen
    documentation["Generated_Plots"] = plot_paths

    # ========== VERBESSERUNGSBERECHNUNG ==========

    # Prozentuale Verbesserung berechnen (korrigiert für DataPath-Struktur)
    improvement_results = []
    datasets = df['DataSet'].unique()
    datapaths = df['DataPath'].unique()

    # Referenzmodell (erstes Modell) für Vergleiche
    reference_model = df['Model'].iloc[0]

    for dataset in datasets:
        for datapath in datapaths:
            df_subset = df[(df['DataSet'] == dataset) & (df['DataPath'] == datapath)]

            if len(df_subset) > 1:  # Mindestens 2 Modelle vorhanden
                reference_mse = df_subset[df_subset['Model'] == reference_model]['MAE'].values

                if len(reference_mse) > 0:
                    reference_mse = reference_mse[0]

                    for _, row in df_subset.iterrows():
                        if row['Model'] != reference_model:
                            improvement = (reference_mse - row['MAE']) / reference_mse * 100
                            improvement_results.append([
                                dataset, datapath, reference_model, row['Model'], improvement
                            ])

    # ========== ERGEBNISSE SPEICHERN ==========

    print("\n===== Modellvergleichsergebnisse =====")
    results_file = os.path.join(results_dir, 'Results.txt')

    with open(results_file, 'w', encoding='utf-8') as f:
        # Hauptergebnisse
        f.write("EXPERIMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'DataSet':<20} | {'DataPath':<15} | {'Model':<15} | {'MAE':<10} | {'StdDev':<10}\n")
        f.write("-" * 80 + "\n")

        for row in results:
            print(f"{row[0]:<20} | {row[1]:<15} | {row[2]:<15} | {row[3]:.6f} | {row[4]:.6f}")
            f.write(f"{row[0]:<20} | {row[1]:<15} | {row[2]:<15} | {row[3]:.6f} | {row[4]:.6f}\n")

        # Verbesserungen
        if improvement_results:
            f.write(f"\n\nMODEL IMPROVEMENTS (vs {reference_model})\n")
            f.write("=" * 80 + "\n")
            f.write(
                f"{'DataSet':<20} | {'DataPath':<15} | {'Reference':<12} | {'Compared':<12} | {'Improvement %':<12}\n")
            f.write("-" * 80 + "\n")

            for row in improvement_results:
                improvement_str = f"{row[4]:+.2f}%"
                print(f"{row[0]:<20} | {row[1]:<15} | {row[2]:<12} | {row[3]:<12} | {improvement_str:<12}")
                f.write(f"{row[0]:<20} | {row[1]:<15} | {row[2]:<12} | {row[3]:<12} | {improvement_str:<12}\n")

        # Plot-Übersicht
        f.write(f"\n\nGENERATED PLOTS\n")
        f.write("=" * 40 + "\n")
        for plot_type, paths in plot_paths.items():
            f.write(f"{plot_type.title()}: {len(paths)} plots\n")
            for path in paths:
                f.write(f"  - {os.path.basename(path)}\n")

    # ========== ERWEITERTE DOKUMENTATION ==========

    documentation["Results"] = {
        #"Model_Comparison": results,
        "Improvement_Analysis": improvement_results,
        "Summary_Statistics": {
            "Total_Experiments": len(results),
            "Datasets_Tested": len(datasets),
            "DataPaths_per_Dataset": {
                dataset: len(df[df['DataSet'] == dataset]['DataPath'].unique())
                for dataset in datasets
            },
            "Models_Compared": df['Model'].unique().tolist(),
            "Best_Model_Overall": df.loc[df['MAE'].idxmin(), 'Model'],
            "Worst_Model_Overall": df.loc[df['MAE'].idxmax(), 'Model']
        }
    }

    # JSON-Dokumentation speichern
    documentation_file = os.path.join(results_dir, 'documentation.json')
    with open(documentation_file, 'w', encoding='utf-8') as json_file:
        json.dump(documentation, json_file, indent=4, ensure_ascii=False)

    # CSV-Export der Ergebniss Übersicht
    csv_file = os.path.join(results_dir, 'results.csv')
    df_csv = df[["DataSet", "DataPath", "Model", "MAE", "StdDev", "MAE_Ensemble"]]
    df_csv.to_csv(csv_file, index=False)
    # CSV-Export der Daten
    hexp.save_detailed_csv(df, results_dir)

    if improvement_results:
        improvement_df = pd.DataFrame(improvement_results,
                                      columns=["DataSet", "DataPath", "Reference_Model",
                                               "Compared_Model", "Improvement_Percent"])
        improvement_csv = os.path.join(results_dir, 'improvements.csv')
        improvement_df.to_csv(improvement_csv, index=False)

    print(f"\n===== Ergebnisse gespeichert =====")
    print(f"Hauptverzeichnis: {results_dir}")
    print(f"Dokumentation: documentation.json, Results.txt")
    print(f"CSV-Dateien: results.csv" + (", improvements.csv" if improvement_results else ""))
    print(f"Plots: {sum(len(paths) for paths in plot_paths.values())} Dateien im plots/ Unterverzeichnis")

    return {
        'results_dir': results_dir,
        'results': results,
        'improvements': improvement_results,
        'documentation': documentation,
        'plot_paths': plot_paths
    }

def start_experiment_for(model_str = 'NN'):
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 1000
    NUMBEROFMODELS = 10 # Bei RF mit festem random state nicht sinvoll

    dataSet = hdata.DataClass_ST_Plate_Notch
    dataclass = copy.copy(dataSet)

    if model_str == 'RF':
        #Random Forest
        search_space = {
            'n_estimators': (5, 500),
            'min_samples_split': (2, 500),
            'min_samples_leaf': (1, 500),
            'max_features': (None, 500),
            'max_depth': (None, 500),
        }
        model = mrf.RandomForestModel()
        prunning = True
        NUMBEROFEPOCHS = 1
    if model_str == 'ET':
        #Random Forest
        search_space = {
            'n_estimators': (5, 500),
            'min_samples_split': (2, 500),
            'min_samples_leaf': (1, 500),
            'max_features': (None, 500),
            'max_depth': (None, 500),
        }
        model = mrf.ExtraTreesModel()
        prunning = True
        NUMBEROFEPOCHS = 1
    elif model_str == 'NN':
        search_space = {
            'n_hidden_size': (1, 100), #250
            'n_hidden_layers': (1, 50), #250
            'learning_rate': (0.001, 0.1),
            'activation': ['ReLU', 'Sigmoid', 'Tanh', 'ELU'],
            'optimizer_type': ['adam', 'sgd', 'quasi_newton'],
        }
        model = mnn.Net()
        prunning = True
    elif model_str == 'RNN':
        search_space = {
            'n_hidden_size': (1, 100), #250
            'n_hidden_layers': (1, 50), #250
            'learning_rate': (0.001, 0.1),
            'activation': ['ReLU', 'Sigmoid', 'Tanh', 'ELU'],
            'optimizer_type': ['adam', 'sgd', 'quasi_newton'],
        }
        model = mnn.RNN()
        dataclass.add_padding = True
        prunning = True
    elif model_str == 'LSTM':
        search_space = {
            'n_hidden_size': (1, 100),
            'n_hidden_layers': (1, 50),
            'learning_rate': (0.001, 0.1),
            'activation': ['ReLU', 'Sigmoid', 'Tanh', 'ELU'],
            'optimizer_type': ['adam', 'sgd', 'quasi_newton'],
        }
        model = mnn.LSTM()
        dataclass.add_padding = True
        prunning = True
    elif model_str == 'GRU':
        search_space = {
            'n_hidden_size': (1, 100),
            'n_hidden_layers': (1, 50),
            'learning_rate': (0.001, 0.1),
            'activation': ['ReLU', 'Sigmoid', 'Tanh', 'ELU'],
            'optimizer_type': ['adam', 'sgd', 'quasi_newton'],
        }
        model = mnn.GRU()
        dataclass.add_padding = True
        prunning = True
    elif model_str == 'PiRNNFriction':
        search_space = {
            'n_hidden_size': (1, 100),
            'n_hidden_layers': (1, 10),
            'learning_rate': (0.01, 0.9),
            'activation': ['ReLU', 'Sigmoid', 'Tanh', 'ELU'],
            'optimizer_type': ['adam', 'quasi_newton'],
            'penalty_weight': (1, 100),
            'penalty_weight_2': (0.0, 2.0),

        }
        model = mnn.PiRNNFriction()
        dataclass.add_sign_hold = True
        dataclass.add_padding = True
        prunning = True
    else:
        throw_error('string is not a valid model')

    # Run the experiment
    run_experiment([dataclass], [model], [search_space],
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS, NUMBEROFTRIALS=NUMBEROFTRIALS,
                        plot_types=['heatmap', 'prediction_overview'], experiment_name=model.name, prunning=prunning)

if __name__ == "__main__":
    start_experiment_for('RNN')