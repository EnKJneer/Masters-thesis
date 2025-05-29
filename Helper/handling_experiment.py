import copy
import json
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime

import Helper.handling_data as hdata
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf

HEADER = ["DataSet", "DataPath", "Model", "MAE", "StdDev", "MAE_Ensemble", "Predictions", "GroundTruth"]

class BasePlotter(ABC):
    """Abstrakte Basisklasse für verschiedene Plot-Typen"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def create_plots(self, df: pd.DataFrame, **kwargs):
        """Erstellt Plots basierend auf dem übergebenen DataFrame"""
        pass

    def save_plot(self, fig, filename: str):
        """Speichert einen Plot"""
        plot_path = os.path.join(self.output_dir, filename)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return plot_path

class DatasetGroupedPlotter(BasePlotter):
    """Erstellt einen Plot pro Dataset (gruppiert alle DataPaths eines Datasets)"""

    def create_plots(self, df: pd.DataFrame, **kwargs):
        datasets = sorted(df['DataSet'].unique())
        models = df['Model'].unique()

        plot_paths = []

        for dataset in datasets:
            fig, ax = plt.subplots(figsize=(12, 6))
            df_dataset = df[df['DataSet'] == dataset]

            # Gruppiere nach DataPath innerhalb des Datasets
            datapaths = sorted(df_dataset['DataPath'].unique())

            x = np.arange(len(datapaths))
            bar_width = 0.8 / len(models)

            for i, model in enumerate(models):
                df_model = df_dataset[df_dataset['Model'] == model]

                # Stelle sicher, dass alle DataPaths vertreten sind
                df_model_indexed = df_model.set_index('DataPath').reindex(datapaths).reset_index()

                y = df_model_indexed['MAE'].values
                yerr = df_model_indexed['StdDev'].values

                x_pos = x + i * bar_width
                bars = ax.bar(x_pos, y, width=bar_width, label=model, yerr=yerr, capsize=4)

                # Text über jedem Balken
                for bar, value in zip(bars, y):
                    if not np.isnan(value):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01 * np.nanmax(y),
                                f'{value:.3f}', ha='center', va='bottom', fontsize=8)

            ax.set_title(f'Model Comparison for Dataset: {dataset}')
            ax.set_xlabel('Data Paths')
            ax.set_ylabel('MAE')
            ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
            ax.set_xticklabels(datapaths, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = f'dataset_comparison_{dataset.replace(" ", "_")}.png'
            plot_path = self.save_plot(fig, filename)
            plot_paths.append(plot_path)

        return plot_paths

class DataPathGroupedPlotter(BasePlotter):
    """Erstellt einen Plot pro DataPath (vergleicht alle Datasets für einen DataPath)"""

    def create_plots(self, df: pd.DataFrame, **kwargs):
        datapaths = sorted(df['DataPath'].unique())
        models = df['Model'].unique()

        plot_paths = []

        for datapath in datapaths:
            fig, ax = plt.subplots(figsize=(12, 6))
            df_datapath = df[df['DataPath'] == datapath]

            datasets = sorted(df_datapath['DataSet'].unique())

            x = np.arange(len(datasets))
            bar_width = 0.8 / len(models)

            for i, model in enumerate(models):
                df_model = df_datapath[df_datapath['Model'] == model]

                # Stelle sicher, dass alle Datasets vertreten sind
                df_model_indexed = df_model.set_index('DataSet').reindex(datasets).reset_index()

                y = df_model_indexed['MAE'].values
                yerr = df_model_indexed['StdDev'].values

                x_pos = x + i * bar_width
                bars = ax.bar(x_pos, y, width=bar_width, label=model, yerr=yerr, capsize=4)

                # Text über jedem Balken
                for bar, value in zip(bars, y):
                    if not np.isnan(value):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01 * np.nanmax(y),
                                f'{value:.3f}', ha='center', va='bottom', fontsize=8)

            ax.set_title(f'Model Comparison for Data Path: {datapath}')
            ax.set_xlabel('Datasets')
            ax.set_ylabel('MAE')
            ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
            ax.set_xticklabels(datasets, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            filename = f'datapath_comparison_{datapath.replace(" ", "_").replace(".csv", "")}.png'
            plot_path = self.save_plot(fig, filename)
            plot_paths.append(plot_path)

        return plot_paths

class ModelComparisonPlotter(BasePlotter):
    """Erstellt eine Übersichtsplot aller Modelle über alle Dataset/DataPath Kombinationen"""

    def create_plots(self, df: pd.DataFrame, **kwargs):
        """
        # Debug: DataFrame-Struktur prüfen
        print("DEBUG - DataFrame Spalten:", df.columns.tolist())
        print("DEBUG - Erste Zeilen:")
        print(df.head())
        print("DEBUG - Modelle gefunden:", df['Model'].unique())
        """
        models = df['Model'].unique()

        # Erstelle eine kombinierte Kategorie aus DataSet und DataPath
        df = df.copy()  # Kopie erstellen um Original nicht zu verändern
        df['Dataset_Path'] = df['DataSet'].astype(str) + '_' + df['DataPath'].astype(str).str.replace('.csv', '')
        categories = sorted(df['Dataset_Path'].unique())

        fig, ax = plt.subplots(figsize=(15, 8))

        x = np.arange(len(categories))
        bar_width = 0.8 / len(models) if len(models) > 0 else 0.8

        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))  # Eindeutige Farben

        for i, model in enumerate(models):
            print(f"DEBUG - Verarbeite Modell: {model} (Typ: {type(model)})")

            df_model = df[df['Model'] == model].copy()

            # Stelle sicher, dass alle Kategorien vertreten sind
            df_model = df_model.set_index('Dataset_Path').reindex(categories).reset_index()

            y = pd.to_numeric(df_model['MAE'], errors='coerce').values
            yerr = pd.to_numeric(df_model['StdDev'], errors='coerce').values

            x_pos = x + i * bar_width
            bars = ax.bar(x_pos, y, width=bar_width, label=str(model),
                          yerr=yerr, capsize=4, color=colors[i], alpha=0.8)

            # Text über jedem Balken (nur bei wenigen Kategorien)
            if len(categories) <= 10:
                for bar, value in zip(bars, y):
                    if not np.isnan(value) and value > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                height + 0.01 * np.nanmax(y[y > 0]) if np.any(y > 0) else height + 0.01,
                                f'{value:.3f}', ha='center', va='bottom', fontsize=7)

        ax.set_title('Complete Model Comparison Overview')
        ax.set_xlabel('Dataset_DataPath Combinations')
        ax.set_ylabel('MAE')
        ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
        ax.set_xticklabels(categories, rotation=90, ha='right')
        ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = 'complete_model_comparison.png'
        plot_path = self.save_plot(fig, filename)

        return [plot_path]

class HeatmapPlotter(BasePlotter):
    """Erstellt eine Heatmap der MAE-Werte"""

    def create_plots(self, df: pd.DataFrame, **kwargs):
        # Pivot-Tabelle erstellen
        df['Dataset_Path'] = df['DataSet'] + '_' + df['DataPath'].str.replace('.csv', '')
        pivot_df = df.pivot_table(values='MAE', index='Dataset_Path', columns='Model', aggfunc='mean')

        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_df.index) * 0.5)))

        im = ax.imshow(pivot_df.values, cmap='RdYlBu_r', aspect='auto')

        # Achsenbeschriftungen
        ax.set_xticks(np.arange(len(pivot_df.columns)))
        ax.set_yticks(np.arange(len(pivot_df.index)))
        ax.set_xticklabels(pivot_df.columns)
        ax.set_yticklabels(pivot_df.index)

        # Rotiere die x-Achsenbeschriftungen
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Werte in die Zellen schreiben
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                value = pivot_df.iloc[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.3f}', ha="center", va="center",
                                   color="white" if value > pivot_df.values.mean() else "black")

        ax.set_title("MAE Heatmap: Models vs Dataset_DataPath")
        fig.colorbar(im, ax=ax, label='MAE')
        plt.tight_layout()

        filename = 'mae_heatmap.png'
        plot_path = self.save_plot(fig, filename)

        return [plot_path]
class HeatmapEnsemblePlotter(BasePlotter):
    """Erstellt eine Heatmap der MAE-Werte"""

    def create_plots(self, df: pd.DataFrame, **kwargs):
        # Pivot-Tabelle erstellen
        df['Dataset_Path'] = df['DataSet'] + '_' + df['DataPath'].str.replace('.csv', '')
        pivot_df = df.pivot_table(values='MAE_Ensemble', index='Dataset_Path', columns='Model', aggfunc='mean')

        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_df.index) * 0.5)))

        im = ax.imshow(pivot_df.values, cmap='RdYlBu_r', aspect='auto')

        # Achsenbeschriftungen
        ax.set_xticks(np.arange(len(pivot_df.columns)))
        ax.set_yticks(np.arange(len(pivot_df.index)))
        ax.set_xticklabels(pivot_df.columns)
        ax.set_yticklabels(pivot_df.index)

        # Rotiere die x-Achsenbeschriftungen
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Werte in die Zellen schreiben
        for i in range(len(pivot_df.index)):
            for j in range(len(pivot_df.columns)):
                value = pivot_df.iloc[i, j]
                if not np.isnan(value):
                    text = ax.text(j, i, f'{value:.3f}', ha="center", va="center",
                                   color="white" if value > pivot_df.values.mean() else "black")

        ax.set_title("MAE_Ensemble Heatmap: Models vs Dataset_DataPath")
        fig.colorbar(im, ax=ax, label='MAE_Ensemble')
        plt.tight_layout()

        filename = 'mae_ensemble_heatmap.png'
        plot_path = self.save_plot(fig, filename)

        return [plot_path]

class PredictionPlotter(BasePlotter):
    """Erstellt Plots mit GroundTruth und Vorhersagen für jeden DataPath"""

    def create_plots(self, df: pd.DataFrame, **kwargs):
        plot_paths = []
        datapaths = df['DataPath'].unique()

        for datapath in datapaths:
            df_subset = df[df['DataPath'] == datapath]
            datasets = df_subset['DataSet'].unique()
            models = df_subset['Model'].unique()
            ground_truth = df_subset['GroundTruth'].iloc[0]

            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot GroundTruth
            ax.plot(ground_truth, label='GroundTruth', color='black', linewidth=2)

            for dataset in datasets:
                # Plot Vorhersagen für jedes Modell
                for model in models:
                    row = df_subset[
                        (df_subset['Model']   == model) &
                        (df_subset['DataSet'] == dataset)
                    ].iloc[0]

                    # angenommen: hier steckt ein Array der Form (n_runs, time)
                    preds_list = np.array(row['Predictions'])

                    # Mittelwert und Standardabweichung über die Runs
                    mean_pred = preds_list.mean(axis=0)
                    std_pred  = preds_list.std(axis=0)

                    t = np.arange(len(mean_pred))
                    # Konfidenzband
                    ax.fill_between(
                        t,
                        mean_pred - std_pred,
                        mean_pred + std_pred,
                        alpha=0.3
                    )
                    # Mittelwertkurve
                    ax.plot(
                        t,
                        mean_pred,
                        label=f"{model} ({dataset})"
                    )

            ax.set_title(f"Predictions vs GroundTruth for {datapath}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            plt.tight_layout()

            filename = f"predictions_vs_groundtruth_{datapath.replace('/', '_')}.png"
            plot_path = self.save_plot(fig, filename)
            plot_paths.append(plot_path)

        return plot_paths

class PredictionPlotterPerModel(BasePlotter):
    """Erstellt Plots mit GroundTruth und Vorhersagen für jeden DataPath und jedes Modell"""

    def create_plots(self, df: pd.DataFrame, **kwargs):
        plot_paths = []
        datapaths = df['DataPath'].unique()

        for datapath in datapaths:
            df_subset = df[df['DataPath'] == datapath]
            datasets = df_subset['DataSet'].unique()
            models = df_subset['Model'].unique()
            ground_truth = df_subset['GroundTruth'].iloc[0]

            for model in models:
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot GroundTruth
                ax.plot(ground_truth, label='GroundTruth', color='black', linewidth=2)

                # Plot Vorhersagen für das aktuelle Modell und jedes Dataset
                for dataset in datasets:
                    predictions_data = df_subset[(df_subset['Model'] == model) & (df_subset['DataSet'] == dataset)]
                    if not predictions_data.empty:
                        row = df_subset[
                            (df_subset['Model'] == model) &
                            (df_subset['DataSet'] == dataset)
                            ].iloc[0]

                        # angenommen: hier steckt ein Array der Form (n_runs, time)
                        preds_list = np.array(row['Predictions'])

                        # Mittelwert und Standardabweichung über die Runs
                        mean_pred = preds_list.mean(axis=0)
                        std_pred = preds_list.std(axis=0)

                        t = np.arange(len(mean_pred))
                        # Konfidenzband
                        ax.fill_between(
                            t,
                            mean_pred - std_pred,
                            mean_pred + std_pred,
                            alpha=0.3
                        )
                        # Mittelwertkurve
                        ax.plot(
                            t,
                            mean_pred,
                            label=f"{model} ({dataset})"
                        )
                    else:
                        print(f"No predictions found for model {model} and dataset {dataset}")

                ax.set_title(f"Predictions vs GroundTruth for {datapath} and {model}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.legend()
                plt.tight_layout()

                filename = f"predictions_vs_groundtruth_{datapath.replace('/', '_')}_{model}.png"
                plot_path = self.save_plot(fig, filename)
                plot_paths.append(plot_path)

        return plot_paths

class PredictionPlotterPerDataset(BasePlotter):
    """Erstellt Plots mit GroundTruth und Vorhersagen für jeden DataPath und jedes DataSet"""

    def create_plots(self, df: pd.DataFrame, **kwargs):
        plot_paths = []
        datasets = df['DataSet'].unique()

        for dataset in datasets:
            df_subset = df[df['DataSet'] == dataset]
            datapaths = df_subset['DataPath'].unique()

            for datapath in datapaths:
                df_datapath = df_subset[df_subset['DataPath'] == datapath]
                models = df_datapath['Model'].unique()
                ground_truth = df_datapath['GroundTruth'].iloc[0]

                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot GroundTruth
                ax.plot(ground_truth, label='GroundTruth', color='black', linewidth=2)

                # Plot Vorhersagen für jedes Modell
                for model in models:
                    predictions = df_datapath[df_datapath['Model'] == model]['Prediction'].iloc[0]
                    ax.plot(predictions, label=model)

                ax.set_title(f"Predictions vs GroundTruth for {dataset} and {datapath}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.legend()
                plt.tight_layout()

                filename = f"predictions_vs_groundtruth_{dataset}_{datapath.replace('/', '_')}.png"
                plot_path = self.save_plot(fig, filename)
                plot_paths.append(plot_path)

        return plot_paths

class PlotManager:
    """Manager-Klasse zum Verwalten verschiedener Plotter"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.plotters = {}

    def register_plotter(self, name: str, plotter: BasePlotter):
        """Registriert einen neuen Plotter"""
        self.plotters[name] = plotter

    def create_all_plots(self, df: pd.DataFrame, selected_plotters: Optional[List[str]] = None):
        """Erstellt alle oder ausgewählte Plots"""
        if selected_plotters is None:
            selected_plotters = list(self.plotters.keys())

        all_plot_paths = {}

        for plotter_name in selected_plotters:
            if plotter_name in self.plotters:
                try:
                    plot_paths = self.plotters[plotter_name].create_plots(df)
                    all_plot_paths[plotter_name] = plot_paths
                    print(f"✓ {plotter_name}: {len(plot_paths)} plots created")
                except Exception as e:
                    print(f"✗ Error creating {plotter_name}: {str(e)}")

        return all_plot_paths

# Beispiel für die Integration in Ihre run_experiment Funktion:
def create_plots_modular(results_dir: str, results: List, plot_types: List[str] = None):
    """
    Erstellt Plots basierend auf den Ergebnissen mit der modularen Plot-Architektur

    Args:
        results_dir: Verzeichnis für die Plots
        results: Liste der Ergebnisse im Format [DataSet, DataPath, Model, MAE, StdDev]
        plot_types: Liste der gewünschten Plot-Typen ['dataset', 'datapath', 'overview', 'heatmap']
    """

    if plot_types is None:
        plot_types = ['heatmap', 'overview', 'prediction_overview']

    # DataFrame erstellen mit expliziter Validierung
    print("DEBUG - Erstelle DataFrame aus results:")
    print(f"Anzahl results: {len(results)}")
    if len(results) > 0:
        print(f"Beispiel result: {results[0]}")
        print(f"Länge erstes Element: {len(results[0])}")

    df = pd.DataFrame(results, columns=HEADER)

    # DataFrame validieren
    print("\nDEBUG - DataFrame Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"\nUnique Models: {df['Model'].unique()}")
    print(f"Model types: {[type(x) for x in df['Model'].unique()]}")

    # Datentypen korrigieren falls nötig
    df['MAE'] = pd.to_numeric(df['MAE'], errors='coerce')
    df['StdDev'] = pd.to_numeric(df['StdDev'], errors='coerce')
    df['Model'] = df['Model'].astype(str)
    df['DataSet'] = df['DataSet'].astype(str)
    df['DataPath'] = df['DataPath'].astype(str)

    print(f"\nNach Typkonvertierung:")
    print(f"MAE NaN values: {df['MAE'].isna().sum()}")
    print(f"StdDev NaN values: {df['StdDev'].isna().sum()}")

    # Plot-Manager initialisieren
    plots_dir = os.path.join(results_dir, "plots")
    plot_manager = PlotManager(plots_dir)

    # Plotter registrieren
    plot_manager.register_plotter('dataset', DatasetGroupedPlotter(plots_dir))
    plot_manager.register_plotter('datapath', DataPathGroupedPlotter(plots_dir))
    plot_manager.register_plotter('overview', ModelComparisonPlotter(plots_dir))
    plot_manager.register_plotter('heatmap', HeatmapPlotter(plots_dir))
    plot_manager.register_plotter('heatmap_ensemble', HeatmapEnsemblePlotter(plots_dir))
    plot_manager.register_plotter('prediction_overview', PredictionPlotter(plots_dir))
    plot_manager.register_plotter('prediction_model', PredictionPlotterPerModel(plots_dir))
    plot_manager.register_plotter('prediction_dataset', PredictionPlotterPerDataset(plots_dir))

    # Plots erstellen
    plot_paths = plot_manager.create_all_plots(df, plot_types)

    return plot_paths

def debug_results_structure(results: List):
    """Debug-Funktion zur Überprüfung der results-Struktur"""
    print("=== RESULTS STRUCTURE DEBUG ===")
    print(f"Total results: {len(results)}")

    if len(results) == 0:
        print("ERROR: Keine Ergebnisse vorhanden!")
        return

    # Erste paar Einträge analysieren
    for i, result in enumerate(results[:5]):
        print(f"Result {i}: {result}")
        print(f"  Length: {len(result)}")
        print(f"  Types: {[type(x) for x in result]}")

        if len(result) >= 5:
            print(f"  DataSet: '{result[0]}' (type: {type(result[0])})")
            print(f"  DataPath: '{result[1]}' (type: {type(result[1])})")
            print(f"  Model: '{result[2]}' (type: {type(result[2])})")
            print(f"  MAE: '{result[3]}' (type: {type(result[3])})")
            print(f"  StdDev: '{result[4]}' (type: {type(result[4])})")
        print()

    # Überprüfe auf häufige Probleme
    unique_lengths = set(len(result) for result in results)
    if len(unique_lengths) > 1:
        print(f"WARNING: Inconsistent result lengths: {unique_lengths}")

    # Überprüfe Modellnamen
    models = set()
    for result in results:
        if len(result) >= 3:
            models.add(str(result[2]))

    print(f"Unique models found: {sorted(models)}")
    print("=== END DEBUG ===\n")

def run_experiment(dataSets, use_nn_reference, use_rf_reference, models,
                   NUMBEROFEPOCHS=800, NUMBEROFMODELS=10, window_size=10,
                   past_values=2, future_values=2, batched_data=False,
                   n_drop_values=20, patience=5, plot_types=None):

    def calculate_and_store_results(model, data, nn_preds, y_test, df_list_results, results, header_list,
                                    n_drop_values):
        """
        Calculate MAE and standard deviation, and store the results.

        Parameters:
        - model: The model being evaluated.
        - data: The data object containing testing data paths.
        - nn_preds: Predictions from the model.
        - y_test: True values.
        - df_list_results: List to store the results DataFrames.
        - results: List to store the results.
        - header_list: List to store the headers.
        - n_drop_values: Number of values to drop for calculation.
        """
        for j, path in enumerate(data.testing_data_paths):
            name = model.name + "_" + path.replace('.csv', '')

            mse_nn, std_nn, mae_ensemble = hdata.calculate_mae_and_std(nn_preds[j],
                                                         y_test[j].values if isinstance(y_test[j], pd.DataFrame) else
                                                         y_test[j], n_drop_values)

            predictions = []
            for pred in nn_preds[j]:
                predictions.append(pred[:-n_drop_values].tolist())

            # Ergebnisse speichern - FIXED: Correct order [DataSet, DataPath, Model, MAE, StdDev]
            df_list_results[j][name] = np.mean(nn_preds[j], axis=0)
            results.append([
                data.name,  # DataSet
                path.replace('.csv', ''),  # DataPath
                model.name,  # Model
                mse_nn,  # MAE
                std_nn,  # StdDev
                mae_ensemble,
                predictions,
                # np.mean(nn_preds[j], axis=0)[:-n_drop_values].tolist(), # mean prediction as list
                y_test[j].iloc[:-n_drop_values].values.tolist() if isinstance(y_test[j], pd.DataFrame) else y_test[j][:-n_drop_values].tolist()  # ground truth as list
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
            model.name+"_Ensemble": {
                "NUMBEROFMODELS": NUMBEROFMODELS,
                **model.get_documentation()
            },
        }
        meta_information["Models"].append(model_info)

    # Add models to meta information
    for model in models:
        model_info = {
            model.name+"_Ensemble": {
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
    for i, data in enumerate(dataSets):
        print(f"\n===== Verarbeitung: {data.name} =====")
        # Daten laden
        X_train, X_val, X_test, y_train, y_val, y_test = data.load_data(
            past_values, future_values, window_size, keep_separate=batched_data)
        if isinstance(X_test, list):
            df_list_results = [pd.DataFrame() for x in range(len(X_test))]
            header_list = [[] for x in range(len(X_test))]
        else:
            df_list_results = [pd.DataFrame()]
            header_list = []

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
            calculate_and_store_results(model, data, nn_preds, y_test, df_list_results, results, header_list,
                                        n_drop_values)

        if len(reference_models) > 0:
            criterion = reference_models[0].criterion
        else:
            criterion = None
        # Train and test models
        for model in models:
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            for _ in range(NUMBEROFMODELS):
                model.train_model(X_train, y_train, X_val, y_val, n_epochs = NUMBEROFEPOCHS, patience=patience)
                if isinstance(X_test, list):
                    for i, (x, y) in enumerate(zip(X_test, y_test)):
                        mse, pred_nn = model.test_model(x, y)
                        print(f"{model.name}: Test RMAE: {mse}")
                        nn_preds[i].append(pred_nn.flatten())
                else:
                    mse, pred_nn = model.test_model(X_test, y_test)
                    print(f"{model.name}: Test RMAE: {mse}")
                    nn_preds.append(pred_nn.flatten())

            # Fehlerberechnung
            calculate_and_store_results(model, data, nn_preds, y_test, df_list_results, results, header_list,
                                        n_drop_values)

        reference_models = reference_models_copy
        models = models_copy
    #debug_results_structure(results)

    # ========== NEUE MODULARE PLOT-ERSTELLUNG ==========

    # DataFrame mit korrigierter Struktur erstellen
    df = pd.DataFrame(results, columns=HEADER)

    # Modulare Plots erstellen
    print("\n===== Erstelle Plots =====")
    plot_paths = create_plots_modular(results_dir, results, plot_types)

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
            "DataPaths_per_Dataset": {dataset: len(df[df['DataSet'] == dataset]['DataPath'].unique())
                                      for dataset in datasets},
            "Models_Compared": df['Model'].unique().tolist(),
            "Best_Model_Overall": df.loc[df['MAE'].idxmin(), 'Model'],
            "Worst_Model_Overall": df.loc[df['MAE'].idxmax(), 'Model']
        },
        "Predictions": {
            datapath: {
                "MeanPredictions": df[df['DataPath'] == datapath][['Model', 'Predictions']].set_index('Model').to_dict()['Predictions'],
                "GroundTruth": df[df['DataPath'] == datapath]['GroundTruth'].iloc[0]
            }
            for datapath in datapaths
        }
    }

    # JSON-Dokumentation speichern
    documentation_file = os.path.join(results_dir, 'documentation.json')
    with open(documentation_file, 'w', encoding='utf-8') as json_file:
        json.dump(documentation, json_file, indent=4, ensure_ascii=False)

    # CSV-Export für weitere Analysen
    csv_file = os.path.join(results_dir, 'results.csv')
    df.to_csv(csv_file, index=False)

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

def reconstruct_results_from_json(json_file_path):
    """
    Reconstruct the results from the JSON file.

    Parameters:
    - json_file_path: Path to the JSON file.

    Returns:
    - A dictionary containing the reconstructed results.
    """
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        documentation = json.load(json_file)

    results = documentation["Results"]["Model_Comparison"]
    predictions = documentation["Results"]["Predictions"]

    reconstructed_results = {
        "Model_Comparison": results,
        "Predictions": predictions
    }

    return reconstructed_results
