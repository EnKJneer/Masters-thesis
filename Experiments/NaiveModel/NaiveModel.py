import copy
import datetime
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.optimize import curve_fit
from torch import nn
import Helper.handling_data as hdata
import Helper.handling_plots as hplot
import Helper.handling_hyperopt as hyperopt
import Helper.handling_experiment as hexp
import Models.model_neural_net as mnn
import Models.model_physical as mphys
import Models.model_random_forest as mrf
import Models.model_mixture_of_experts as mmix
from datetime import datetime

def linear(x, a, b):
    return a * x + b

def sigmoid(x, a, b, c):
    return a / (1 + np.exp(-(x + b))) + c

def combined_model(x, a1, b1, a2, b2, c2):
    return linear(x[0], a1, b1) + sigmoid(x[1], a2, b2, c2)

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def run_experiment(dataSets, use_nn_reference, use_rf_reference, models,
                   NUMBEROFEPOCHS=800, NUMBEROFMODELS=10, window_size=10,
                   past_values=2, future_values=2, batched_data=False,
                   n_drop_values=20, patience=5, plot_types=None):
    # In calculate_and_store_results Funktion:
    def calculate_and_store_results(model, dataClass, nn_preds, y_test, df_list_results, results, header_list,
                                    n_drop_values, raw_data):  # dataSets Parameter hinzufügen
        """
        Calculate MAE and standard deviation, and store the results.
        """
        for j, path in enumerate(dataClass.testing_data_paths):
            name = model.name + "_" + path.replace('.csv', '')

            mse_nn, std_nn, mae_ensemble = hdata.calculate_mae_and_std(nn_preds[j],
                                                                       y_test[j].values if isinstance(y_test[j],
                                                                                                      pd.DataFrame) else
                                                                       y_test[j], n_drop_values)

            predictions = []
            for pred in nn_preds[j]:
                predictions.append(pred[:-n_drop_values].tolist())

            # Raw data als Dictionary speichern (JSON-serialisierbar)
            raw_data_dict = {
                'columns': raw_data[j].columns.tolist(),
                'data': raw_data[j].iloc[:-n_drop_values].to_dict('records')  # Als Liste von Dictionaries
            }

            # Ergebnisse speichern mit raw_data
            df_list_results[j][name] = np.mean(nn_preds[j], axis=0)
            results.append([
                dataClass.name,  # dataClass
                path.replace('.csv', ''),  # DataPath
                model.name,  # Model
                mse_nn,  # MAE
                std_nn,  # StdDev
                mae_ensemble,
                predictions,
                y_test[j].iloc[:-n_drop_values].values.tolist() if isinstance(y_test[j], pd.DataFrame) else y_test[
                                                                                                                j][
                                                                                                            :-n_drop_values].tolist(),
                raw_data_dict  # RawData als Dictionary
            ])
            header_list[j].append(name)

    if type(dataSets) is not list:
        dataSets = [dataSets]
    if type(models) is not list:
        models = [models]

    # Create directory for results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = os.path.join("Results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Define the meta information structure
    meta_information = {
        "DataSets": [],
        "Models": [],
        "Data_Preprocessing": {
            "window_size": window_size,
            "past_values": past_values,
            "future_values": future_values,
            "batched_data": batched_data,
            "n_drop_values": n_drop_values,
            "NUMBEROFEPOCHS": NUMBEROFEPOCHS,
        }
    }

    # Define reference models
    reference_models = []
    if use_nn_reference:
        model_nn = mnn.get_reference()
        reference_models.append(model_nn)
    if use_rf_reference:
        model_rf = mrf.get_reference()
        reference_models.append(model_rf)

    # Add reference models to meta information
    for model in reference_models:
        model_info = {
            model.name + "_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                **model.get_documentation()
            },
        }
        meta_information["Models"].append(model_info)

    # Add models to meta information
    for model in models:
        model_info = {
            model.name + "_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                **model.get_documentation()
            },
        }
        meta_information["Models"].append(model_info)

    # Add datasets to meta information
    for data_params in dataSets:
        data_info = {
            **data_params.get_documentation()
        }
        meta_information["DataSets"].append(data_info)

    # Save the meta information to a JSON file
    documentation = meta_information

    """ Check if data_params.testing_data_paths is the same """

    """ Prediction """
    results = []
    models_copy = copy.deepcopy(models)
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

        # Train and test models
        for model in models:
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            for _ in range(NUMBEROFMODELS):
                f_x =  np.array(X_train['f_x_sim_1_current'].values)
                v_x =  np.array(X_train['v_x_1_current'].values)
                y =  np.array(y_train.values)
                initial_params_combined = [1, 1, 1, 1, 1]
                params_combined, _ = curve_fit(combined_model, (f_x, v_x), y, p0=initial_params_combined)

                if isinstance(X_test, list):
                    for i, (x, y) in enumerate(zip(X_test, y_test)):
                        f_x = X_test[i]['f_x_sim_1_current']
                        v_x = X_test[i]['v_x_1_current']
                        pred_nn = combined_model((f_x, v_x), *params_combined)
                        mse = calculate_mae(y_test, pred_nn)
                        print(f"{model.name}: Test RMAE: {mse}")
                        nn_preds.append(pred_nn.flatten())
                else:
                    f_x = raw_data['f_x_sim']
                    v_x = raw_data['v_x_sim']
                    pred_nn = combined_model((f_x, v_x), *params_combined)
                    mse = calculate_mae(y_test, pred_nn)
                    print(f"{model.name}: Test RMAE: {mse}")
                    nn_preds.append(pred_nn.flatten())

            # Fehlerberechnung
            calculate_and_store_results(model, dataClass, nn_preds, y_test, df_list_results, results, header_list,
                                        n_drop_values, raw_data)

        # Train and test reference models
        for model in reference_models:
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            for _ in range(NUMBEROFMODELS):
                model.train_model(X_train, y_train, X_val, y_val, NUMBEROFEPOCHS, patience=patience)
                if isinstance(X_test, list):
                    for i, (x, y) in enumerate(zip(X_test, y_test)):
                        mse, pred_nn = model.test_model(x, y)
                        nn_preds[i].append(pred_nn.flatten())
                        print(f"{model.name}: Test MAE: {mse}")
                else:
                    mse, pred_nn = model.test_model(X_test, y_test)
                    nn_preds.append(pred_nn.flatten())
                    print(f"{model.name}: Test MAE: {mse}")

            # Fehlerberechnung
            calculate_and_store_results(model, dataClass, nn_preds, y_test, df_list_results, results, header_list,
                                        n_drop_values, raw_data)



        reference_models = reference_models_copy
        models = models_copy
    # debug_results_structure(results)

    # ========== NEUE MODULARE PLOT-ERSTELLUNG ==========

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
        "Model_Comparison": results,
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
        },
        "Predictions": {
            datapath: {
                "MeanPredictions":
                    df[df['DataPath'] == datapath][['Model', 'Predictions']].set_index('Model').to_dict()[
                        'Predictions'],
                "GroundTruth": df[df['DataPath'] == datapath]['GroundTruth'].iloc[0],
                "RawData": df[df['DataPath'] == datapath]['RawData'].iloc[0]  # RawData hinzufügen
            }
            for datapath in datapaths
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
    save_detailed_csv(df, results_dir)

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


def save_detailed_csv(df, results_dir):
    """
    Speichert detaillierte Daten für jeden DataPath in separaten CSV-Dateien.

    Args:
        df (pd.DataFrame): DataFrame mit den Ergebnissen.
        results_dir (str): Verzeichnis, in dem die CSV-Dateien gespeichert werden sollen.
    """
    # Erstelle ein Unterverzeichnis für die detaillierten CSV-Dateien
    detailed_csv_dir = os.path.join(results_dir, 'Predictions')
    os.makedirs(detailed_csv_dir, exist_ok=True)

    # Iteriere über jeden eindeutigen DataPath
    for datapath in df['DataPath'].unique():
        df_subset = df[df['DataPath'] == datapath]

        # Erstelle einen DataFrame für die Rohdaten
        raw_data = df_subset['RawData'].iloc[0]
        raw_data_df = pd.DataFrame(raw_data['data'], columns=raw_data['columns'])

        # Erstelle einen DataFrame für den Ground Truth
        ground_truth = df_subset['GroundTruth'].iloc[0]
        ground_truth_df = pd.DataFrame(ground_truth, columns=['GroundTruth'])

        # Erstelle einen DataFrame für die mittleren Vorhersagen jeder DataSet-Modell-Kombination
        predictions_dict = {}
        for _, row in df_subset.iterrows():
            predictions = row['Predictions']
            mean_predictions = np.mean(predictions, axis=0)
            predictions_dict[f'{row["DataSet"]}_{row["Model"]}'] = mean_predictions

        # Erstelle einen DataFrame aus dem Wörterbuch der mittleren Vorhersagen
        predictions_df = pd.DataFrame(predictions_dict)

        # Kombiniere die DataFrames
        combined_df = pd.concat([raw_data_df, ground_truth_df, predictions_df], axis=1)

        # Speichere den kombinierten DataFrame in einer CSV-Datei
        csv_file = os.path.join(detailed_csv_dir, f'{datapath.replace("/", "_")}.csv')
        combined_df.to_csv(csv_file, index=False)

    print(f"Detaillierte CSV-Dateien wurden in {detailed_csv_dir} gespeichert.")


if __name__ == "__main__":
    """ Constants """
    NUMBEROFTRIALS = 250
    NUMBEROFEPOCHS = 800
    NUMBEROFMODELS = 2

    window_size = 1
    past_values = 0
    future_values = 0

    #Combined_Gear,Combined_KL
    dataClass_1 = hdata.Combined_Plate_TrainVal
    dataClass_1.window_size = window_size
    dataClass_1.past_values = past_values
    dataClass_1.future_values = future_values

    dataSets_list = [dataClass_1]

    #model_simple = mphys.NaiveModelSimple()
    model = mphys.NaiveModelSigmoid()
    model_2 = mphys.NaiveModel()
    model_erd = mphys.PhysicalModelErdSingleAxis()
    models = [model_2, model] #model_erd

    # Run the experiment
    hexp.run_experiment(dataSets_list, use_nn_reference=True, use_rf_reference=False, models=models,
                        NUMBEROFEPOCHS=NUMBEROFEPOCHS, NUMBEROFMODELS=NUMBEROFMODELS,
                        window_size=window_size, past_values=past_values, future_values=future_values,
                        plot_types=['heatmap', 'prediction_overview'])


