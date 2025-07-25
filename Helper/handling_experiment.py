import copy
import json
import os
import ast
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
import seaborn as sns
from sklearn.metrics import mean_absolute_error

import Helper.handling_data as hdata
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
import Models.model_physical as mphys
from matplotlib.colors import LinearSegmentedColormap

HEADER = ["DataSet", "DataPath", "Model", "MAE", "StdDev", "MAE_Ensemble", "Predictions", "GroundTruth", "RawData"]

def plot_2d_with_color(x_values, y_values, color_values, titel='|error|', label_colour = 'mae', dpi=300, xlabel = 'pos_x', ylabel = 'pos_y'):
    """
    Erstellt einen 2D-Plot mit Linien, deren Farbe basierend auf den color_values bestimmt wird.

    :param x_values: Liste oder Array der x-Werte
    :param y_values: Liste oder Array der y-Werte
    :param color_values: Liste oder Array der Werte, die die Farbe bestimmen
    :param name: Name der Farbskala (Standard: '|error|')
    :param dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
    """
    # Erstellen des Plots mit höherer Auflösung
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)

    # Normalisieren der color_values für den Farbverlauf
    normalized_color_values = (color_values - np.min(color_values)) / (np.max(color_values) - np.min(color_values))

    # Erstellen eines Farbverlaufs basierend auf den color_values
    #for i in range(len(x_values) - 1):
    #    ax.plot(x_values[i:i + 2], y_values[i:i + 2], c=plt.cm.viridis(normalized_color_values[i]))

    # Erstellen eines Streudiagramms, um die Farbskala anzuzeigen
    sc = ax.scatter(x_values, y_values, c=color_values, cmap='viridis', s=5)

    # Hinzufügen einer Farbskala
    plt.colorbar(sc, label=label_colour)

    # Beschriftungen und Titel hinzufügen
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{titel}')

    return fig

class BasePlotter(ABC):
    """Abstrakte Basisklasse für verschiedene Plot-Typen"""

    def __init__(self, output_dir: str, known_material: str = 'S235JR', known_geometry: str = 'Plate'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # KIT-Farben definieren
        self.kit_red = "#B2372C"
        self.kit_green = "#009682"
        self.kit_yellow = "#EEB70D"

        # Benutzerdefinierte Farbpalette erstellen (grün=gut, gelb=mittel, rot=schlecht)
        self.custom_cmap = LinearSegmentedColormap.from_list(
            'kit_colors',
            [self.kit_green, self.kit_yellow, self.kit_red],
            N=256
        )

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

    def parse_filename(self, filename: str):
        """Parst den Dateinamen und extrahiert Material und Geometrie"""
        # Entferne .csv Extension
        name_without_ext = filename.replace('.csv', '')

        # Split nach Unterstrichen
        parts = name_without_ext.split('_')

        print(f"Debug: Parsing '{filename}' -> Parts: {parts}")

        # Für deine spezifischen Dateinamen:
        # AL_2007_T4_Gear_Normal_3.csv -> ['AL', '2007', 'T4', 'Gear', 'Normal', '3']
        # S235JR_Plate_Normal_3.csv -> ['S235JR', 'Plate', 'Normal', '3']

        material = 'Unknown'
        geometry = 'Unknown'

        if len(parts) >= 4:
            # AL_2007_T4 Fall
            if parts[0] == 'AL' and parts[1] == '2007' and parts[2] == 'T4':
                material = 'AL_2007_T4'
                geometry = parts[3]  # Gear oder Plate
            # S235JR Fall
            elif parts[0] == 'S235JR':
                material = 'S235JR'
                geometry = parts[1]  # Gear oder Plate
            else:
                # Fallback: Versuche erste 3 Teile als Material
                potential_material = '_'.join(parts[:3])
                if potential_material in ['AL_2007_T4']:
                    material = potential_material
                    geometry = parts[3]
                else:
                    # Versuche ersten Teil als Material
                    material = parts[0]
                    geometry = parts[1] if len(parts) > 1 else 'Unknown'
        elif len(parts) >= 2:
            material = parts[0]
            geometry = parts[1]

        print(f"Debug: '{filename}' -> Material: '{material}', Geometry: '{geometry}'")
        return material, geometry

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
    """Erstellt eine Heatmap der MAE-Werte mit MAE und Standardabweichung"""

    def create_plots(self, df: pd.DataFrame, **kwargs):
        # Model + DataSet Kombination erstellen
        df['Model_Dataset'] = df['Model'] + '_' + df['DataSet']

        # Pivot-Tabellen für MAE und StdDev erstellen
        pivot_mae = df.pivot_table(values='MAE', index='DataPath', columns='Model_Dataset', aggfunc='mean')
        pivot_std = df.pivot_table(values='StdDev', index='DataPath', columns='Model_Dataset', aggfunc='mean')

        fig, ax = plt.subplots(figsize=(max(10, len(pivot_mae.columns) * 0.8), max(6, len(pivot_mae.index) * 0.5)))

        im = ax.imshow(pivot_mae.values, cmap=self.custom_cmap, aspect='auto')

        # Achsenbeschriftungen
        ax.set_xticks(np.arange(len(pivot_mae.columns)))
        ax.set_yticks(np.arange(len(pivot_mae.index)))
        ax.set_xticklabels(pivot_mae.columns)
        ax.set_yticklabels([path.replace('.csv', '') for path in pivot_mae.index])

        # Rotiere die x-Achsenbeschriftungen
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Werte in die Zellen schreiben (MAE ± StdDev)
        for i in range(len(pivot_mae.index)):
            for j in range(len(pivot_mae.columns)):
                mae_value = pivot_mae.iloc[i, j]
                std_value = pivot_std.iloc[i, j]

                if not np.isnan(mae_value) and not np.isnan(std_value):
                    text = ax.text(j, i, f'{mae_value:.3f}\n±{std_value:.3f}',
                                   ha="center", va="center",
                                   color="white" if mae_value > pivot_mae.values.mean() else "black",
                                   fontsize=9)

        ax.set_title("MAE Heatmap: DataPath vs Model_DataSet")
        ax.set_xlabel("Model + DataSet")
        ax.set_ylabel("DataPath")
        fig.colorbar(im, ax=ax, label='MAE')
        plt.tight_layout()

        filename = 'heatmap_mae_with_std.png'
        plot_path = self.save_plot(fig, filename)

        return [plot_path]

class ModelHeatmapPlotter(BasePlotter):
    """Erstellt separate Heatmaps für jedes ML-Modell"""

    def __init__(self, output_dir: str, known_material: str = 'AL_2007_T4', known_geometry: str = 'Plate'):
        super().__init__(output_dir)
        self.known_material = known_material
        self.known_geometry = known_geometry

        # KIT-Farben definieren
        self.kit_red = "#B2372C"
        self.kit_green = "#009682"
        self.kit_yellow = "#EEB70D"

        # Benutzerdefinierte Farbpalette erstellen (grün=gut, gelb=mittel, rot=schlecht)
        self.custom_cmap = LinearSegmentedColormap.from_list(
            'kit_colors',
            [self.kit_green, self.kit_yellow, self.kit_red],
            N=256
        )

    def parse_filename(self, filename: str):
        """Parst den Dateinamen und extrahiert Material und Geometrie"""
        # Entferne .csv Extension
        name_without_ext = filename.replace('.csv', '')

        # Split nach Unterstrichen
        parts = name_without_ext.split('_')

        print(f"Debug: Parsing '{filename}' -> Parts: {parts}")

        # Für deine spezifischen Dateinamen:
        # AL_2007_T4_Gear_Normal_3.csv -> ['AL', '2007', 'T4', 'Gear', 'Normal', '3']
        # S235JR_Plate_Normal_3.csv -> ['S235JR', 'Plate', 'Normal', '3']

        material = 'Unknown'
        geometry = 'Unknown'

        if len(parts) >= 4:
            # AL_2007_T4 Fall
            if parts[0] == 'AL' and parts[1] == '2007' and parts[2] == 'T4':
                material = 'AL_2007_T4'
                geometry = parts[3]  # Gear oder Plate
            # S235JR Fall
            elif parts[0] == 'S235JR':
                material = 'S235JR'
                geometry = parts[1]  # Gear oder Plate
            else:
                # Fallback: Versuche erste 3 Teile als Material
                potential_material = '_'.join(parts[:3])
                if potential_material in ['AL_2007_T4']:
                    material = potential_material
                    geometry = parts[3]
                else:
                    # Versuche ersten Teil als Material
                    material = parts[0]
                    geometry = parts[1] if len(parts) > 1 else 'Unknown'
        elif len(parts) >= 2:
            material = parts[0]
            geometry = parts[1]

        print(f"Debug: '{filename}' -> Material: '{material}', Geometry: '{geometry}'")
        return material, geometry

    def calculate_mae_for_file(self, file_path: str, model_columns: list):
        """Berechnet MAE für alle Modelle in einer Datei"""
        try:
            df = pd.read_csv(file_path)
            print(f"Debug: Spalten in {os.path.basename(file_path)}: {df.columns.tolist()}")

            mae_results = {}

            for model_col in model_columns:
                if model_col in df.columns and 'GroundTruth' in df.columns:
                    # Entferne NaN-Werte
                    valid_indices = ~(df['GroundTruth'].isna() | df[model_col].isna())
                    valid_count = valid_indices.sum()

                    print(f"Debug: {model_col} - Gültige Werte: {valid_count}/{len(df)}")

                    if valid_count > 0:
                        mae = mean_absolute_error(
                            df.loc[valid_indices, 'GroundTruth'],
                            df.loc[valid_indices, model_col]
                        )
                        mae_results[model_col] = mae
                        print(f"Debug: {model_col} MAE: {mae}")
                    else:
                        mae_results[model_col] = np.nan
                        print(f"Debug: {model_col} - Keine gültigen Werte")
                else:
                    mae_results[model_col] = np.nan
                    missing_cols = []
                    if model_col not in df.columns:
                        missing_cols.append(model_col)
                    if 'GroundTruth' not in df.columns:
                        missing_cols.append('GroundTruth')
                    print(f"Debug: {model_col} - Fehlende Spalten: {missing_cols}")

        except Exception as e:
            print(f"Debug: Fehler beim Lesen von {file_path}: {e}")
            mae_results = {model_col: np.nan for model_col in model_columns}

        return mae_results

    def create_mae_dataframe(self, folder_path: str, model_columns: list):
        """Erstellt einen DataFrame mit MAE-Werten für alle Dateien"""
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        results = []

        print(f"Debug: Gefundene Dateien: {files}")

        for file in files:
            file_path = os.path.join(folder_path, file)
            material, geometry = self.parse_filename(file)
            mae_results = self.calculate_mae_for_file(file_path, model_columns)

            for model, mae in mae_results.items():
                results.append({
                    'Filename': file,
                    'Material': material,
                    'Geometry': geometry,
                    'Model': model,
                    'MAE': mae
                })
                print(f"Debug: {file} -> {material}/{geometry}/{model} -> MAE: {mae}")

        df = pd.DataFrame(results)
        print(f"Debug: MAE DataFrame shape: {df.shape}")
        print(f"Debug: Unique Materials: {df['Material'].unique()}")
        print(f"Debug: Unique Geometries: {df['Geometry'].unique()}")

        return df

    def create_ordered_categories(self, materials: list, geometries: list):
        """Ordnet die Kategorien so, dass known oben links und unknown unten rechts sind"""
        # Material-Reihenfolge: known zuerst, dann alphabetisch
        materials_ordered = []
        if self.known_material in materials:
            materials_ordered.append(self.known_material)
        for mat in sorted(materials):
            if mat != self.known_material:
                materials_ordered.append(mat)

        # Geometrie-Reihenfolge: known zuerst, dann alphabetisch
        geometries_ordered = []
        if self.known_geometry in geometries:
            geometries_ordered.append(self.known_geometry)
        for geo in sorted(geometries):
            if geo != self.known_geometry:
                geometries_ordered.append(geo)

        return materials_ordered, geometries_ordered

    def create_plots(self, folder_path: str, model_columns: list, **kwargs):
        """Erstellt für jedes Modell eine separate Heatmap"""
        # MAE-Daten erstellen
        mae_df = self.create_mae_dataframe(folder_path, model_columns)

        # Unique Materialien und Geometrien ermitteln
        materials = mae_df['Material'].unique()
        geometries = mae_df['Geometry'].unique()

        # Kategorien ordnen
        materials_ordered, geometries_ordered = self.create_ordered_categories(materials, geometries)

        plot_paths = []

        # Für jedes Modell eine separate Heatmap erstellen
        for model in model_columns:
            model_data = mae_df[mae_df['Model'] == model]

            # Pivot-Tabelle für dieses Modell erstellen
            pivot_df = model_data.pivot_table(
                values='MAE',
                index='Material',
                columns='Geometry',
                aggfunc='mean'
            )

            # Reorder basierend auf der gewünschten Reihenfolge
            pivot_df = pivot_df.reindex(
                index=materials_ordered,
                columns=geometries_ordered
            )

            # Heatmap erstellen
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.set_dpi(1200)
            # Maske für NaN-Werte
            mask = pivot_df.isna()

            # Heatmap mit seaborn für bessere Optik
            sns.heatmap(
                pivot_df,
                annot=True,
                fmt='.4f',
                cmap=self.custom_cmap,
                mask=mask,
                cbar_kws={'label': 'MAE'},
                ax=ax,
                square=True,
                vmin=0,  # Minimum-Wert für Colorbar
                linewidths=0.5,  # Linien zwischen Zellen
                linecolor='gray',
                annot_kws={'size': 20, 'weight': 'bold'}  # Größere Schrift für Werte
            )
            model = model.replace('Plate_TrainVal_', '')
            model = model.replace('_', ' ')
            # Titel und Labels mit größerer Schrift
            ax.set_title(f'MAE Heatmap: {model}', fontsize=20, fontweight='bold', pad=20)
            ax.set_xlabel('Geometry', fontsize=20, fontweight='bold')
            ax.set_ylabel('Material', fontsize=20, fontweight='bold')

            # Achsenbeschriftungen vergrößern
            ax.tick_params(axis='both', which='major', labelsize=20)

            # Colorbar-Label vergrößern
            cbar = ax.collections[0].colorbar
            cbar.set_label('MAE', fontsize=20, fontweight='bold')
            cbar.ax.tick_params(labelsize=16)

            plt.tight_layout()

            # Speichern
            filename = f'heatmap_{model.replace(" ", "_")}.png'
            plot_path = self.save_plot(fig, filename)
            plot_paths.append(plot_path)

            print(f"Heatmap für {model} erstellt: {plot_path}")

        return plot_paths

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
            y_min, y_max = float('inf'), -float('inf')
            # Plot GroundTruth
            ax.plot(ground_truth, label='GroundTruth', color='black', linewidth=2)

            for dataset in datasets:
                for model in models:
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

                    # Konfidenzband auf Mittelwert plus/minus vier Standardabweichungen beschränken
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
                    # Bestimme die Minimal- und Maximalwerte für die y-Achse
                    current_min = np.min(mean_pred - 4 * std_pred)
                    current_max = np.max(mean_pred + 4 * std_pred)
                    if current_min < y_min:
                        y_min = current_min
                    if current_max > y_max:
                        y_max = current_max

            if -8 > y_min:
                y_min = -8
            if 8 < y_max:
                y_max = 8

            ax.set_title(f"Predictions vs GroundTruth for {datapath}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.set_ylim(y_min, y_max)
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

class MAEPlotterGeometry(BasePlotter):
    def create_plots(self, df: pd.DataFrame, **kwargs):
        plot_paths = []
        datapaths = df['DataPath'].unique()

        for datapath in datapaths:
            df_subset = df[df['DataPath'] == datapath]

            datasets = df_subset['DataSet'].unique()
            models = df_subset['Model'].unique()
            ground_truth = np.array(df_subset['GroundTruth'].iloc[0])

            # Plot Vorhersagen für das aktuelle Modell und jedes Dataset
            for dataset in datasets:
                for model in models:
                    predictions_data = df_subset[(df_subset['Model'] == model) & (df_subset['DataSet'] == dataset)]
                    if not predictions_data.empty:
                        row = df_subset[
                            (df_subset['Model'] == model) &
                            (df_subset['DataSet'] == dataset)
                            ].iloc[0]

                        preds_list = np.array(row['Predictions'])
                        raw_data_dict = row['RawData']
                        data_geometry = reconstruct_raw_data(raw_data_dict)

                        # Überprüfen, ob benötigte Spalten existieren
                        if 'pos_x' not in data_geometry.columns or 'pos_y' not in data_geometry.columns:
                            print(
                                f"Required columns 'pos_x' or 'pos_y' not found in RawData for {datapath}, {dataset}, {model}")
                            print(f"Available columns: {data_geometry.columns.tolist()}")
                            continue

                        # Mittelwert und Standardabweichung über die Runs
                        mean_pred = preds_list.mean(axis=0)
                        std_pred = preds_list.std(axis=0)
                        mae = (mean_pred.squeeze() - ground_truth.squeeze())

                        # Längen-Check für Sicherheit
                        min_length = min(len(data_geometry), len(mae))
                        if min_length < len(data_geometry):
                            print(f"Warning: Truncating data to length {min_length} for {datapath}, {dataset}, {model}")
                            data_geometry = data_geometry.iloc[:min_length]
                            mae = mae[:min_length]

                        # Plot mit Farbe
                        fig = plot_2d_with_color(
                            data_geometry['pos_x'].values,
                            data_geometry['pos_y'].values,
                            mae,
                            f'{datapath} {dataset} {model} mae'
                        )
                        filename = f"MAE_Geometry_{datapath.replace('/', '_')}_{dataset}_{model}.png"
                        plot_path = self.save_plot(fig, filename)
                        plot_paths.append(plot_path)
                    else:
                        print(f"No predictions found for model {model} and dataset {dataset}")
        return plot_paths

class MAEPlotterForce(BasePlotter):
    def create_plots(self, df: pd.DataFrame, **kwargs):
        plot_paths = []
        datapaths = df['DataPath'].unique()

        for datapath in datapaths:
            df_subset = df[df['DataPath'] == datapath]

            datasets = df_subset['DataSet'].unique()
            models = df_subset['Model'].unique()
            ground_truth = np.array(df_subset['GroundTruth'].iloc[0])

            # Plot Vorhersagen für das aktuelle Modell und jedes Dataset
            for dataset in datasets:
                for model in models:
                    predictions_data = df_subset[(df_subset['Model'] == model) & (df_subset['DataSet'] == dataset)]
                    if not predictions_data.empty:
                        row = df_subset[
                            (df_subset['Model'] == model) &
                            (df_subset['DataSet'] == dataset)
                            ].iloc[0]

                        preds_list = np.array(row['Predictions'])
                        raw_data_dict = row['RawData']
                        data_geometry = reconstruct_raw_data(raw_data_dict)

                        key_x_data = 'f_x_sim'
                        key_y_data = 'f_y_sim'
                        # Überprüfen, ob benötigte Spalten existieren
                        if key_x_data not in data_geometry.columns or key_y_data not in data_geometry.columns:
                            print(
                                f"Required columns '{key_x_data}' or '{key_y_data}' not found in RawData for {datapath}, {dataset}, {model}")
                            print(f"Available columns: {data_geometry.columns.tolist()}")
                            continue

                        # Mittelwert und Standardabweichung über die Runs
                        mean_pred = preds_list.mean(axis=0)
                        std_pred = preds_list.std(axis=0)
                        mae = np.abs(mean_pred.squeeze() - ground_truth.squeeze())

                        # Längen-Check für Sicherheit
                        min_length = min(len(data_geometry), len(mae))
                        if min_length < len(data_geometry):
                            print(f"Warning: Truncating data to length {min_length} for {datapath}, {dataset}, {model}")
                            data_geometry = data_geometry.iloc[:min_length]
                            mae = mae[:min_length]

                        # Plot mit Farbe
                        fig = plot_2d_with_color(
                            data_geometry[key_x_data].values,
                            data_geometry[key_y_data].values,
                            mae,
                            f'{datapath} {dataset} {model} mae',
                            xlabel=key_x_data,
                            ylabel=key_y_data
                        )
                        filename = f"MAE_Force_{datapath.replace('/', '_')}_{dataset}_{model}.png"
                        plot_path = self.save_plot(fig, filename)
                        plot_paths.append(plot_path)
                    else:
                        print(f"No predictions found for model {model} and dataset {dataset}")
        return plot_paths

class MAEPlotterMRR(BasePlotter):
    def create_plots(self, df: pd.DataFrame, **kwargs):
        plot_paths = []
        datapaths = df['DataPath'].unique()

        for datapath in datapaths:
            df_subset = df[df['DataPath'] == datapath]

            datasets = df_subset['DataSet'].unique()
            models = df_subset['Model'].unique()
            ground_truth = np.array(df_subset['GroundTruth'].iloc[0])

            # Plot Vorhersagen für das aktuelle Modell und jedes Dataset
            for dataset in datasets:
                for model in models:
                    predictions_data = df_subset[(df_subset['Model'] == model) & (df_subset['DataSet'] == dataset)]
                    if not predictions_data.empty:
                        row = df_subset[
                            (df_subset['Model'] == model) &
                            (df_subset['DataSet'] == dataset)
                            ].iloc[0]

                        preds_list = np.array(row['Predictions'])
                        raw_data_dict = row['RawData']
                        data_geometry = reconstruct_raw_data(raw_data_dict)

                        key_x_data = 'materialremoved_sim'
                        # Überprüfen, ob benötigte Spalten existieren
                        if key_x_data not in data_geometry.columns:
                            print(
                                f"Required columns '{key_x_data}' not found in RawData for {datapath}, {dataset}, {model}")
                            print(f"Available columns: {data_geometry.columns.tolist()}")
                            continue

                        # Mittelwert und Standardabweichung über die Runs
                        mean_pred = preds_list.mean(axis=0)
                        std_pred = preds_list.std(axis=0)
                        mae = np.abs(mean_pred.squeeze() - ground_truth.squeeze())

                        # Längen-Check für Sicherheit
                        min_length = min(len(data_geometry), len(mae))
                        if min_length < len(data_geometry):
                            print(f"Warning: Truncating data to length {min_length} for {datapath}, {dataset}, {model}")
                            data_geometry = data_geometry.iloc[:min_length]
                            mae = mae[:min_length]

                        fig, ax = plt.subplots(figsize=(10, 6))

                        mrr = data_geometry[key_x_data].values
                        # Mittelwertkurve
                        sc = ax.scatter(
                            mrr,
                            mae,
                            label=f"{model} ({dataset})", s=5,
                            c=ground_truth, cmap='viridis'
                        )

                        ax.set_title(f"MAE vs MRR for {datapath} and {model}")
                        ax.set_xlabel("MRR")
                        ax.set_ylabel("MAE")
                        ax.legend()
                        plt.tight_layout()

                        # Hinzufügen einer Farbskala
                        plt.colorbar(sc, label='ground truth')

                        filename = f"MAE_MRR_{datapath.replace('/', '_')}_{dataset}_{model}.png"
                        plot_path = self.save_plot(fig, filename)
                        plot_paths.append(plot_path)
                    else:
                        print(f"No predictions found for model {model} and dataset {dataset}")
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
def create_plots_modular(results_dir: str, results: List, plot_types: List[str] = None, DEBUG = False):
    """
    Erstellt Plots basierend auf den Ergebnissen mit der modularen Plot-Architektur

    Args:
        results_dir: Verzeichnis für die Plots
        results: Liste der Ergebnisse im Format [DataSet, DataPath, Model, MAE, StdDev]
        plot_types: Liste der gewünschten Plot-Typen ['dataset', 'datapath', 'overview', 'heatmap']
    """

    if plot_types is None:
        plot_types = ['heatmap', 'overview', 'prediction_overview']

    if DEBUG:
        # DataFrame erstellen mit expliziter Validierung
        print("DEBUG - Erstelle DataFrame aus results:")
        print(f"Anzahl results: {len(results)}")
        if len(results) > 0:
            print(f"Beispiel result: {results[0]}")
            print(f"Länge erstes Element: {len(results[0])}")

    df = pd.DataFrame(results, columns=HEADER)

    if DEBUG:
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

    if DEBUG:
        print(f"\nNach Typkonvertierung:")
        print(f"MAE NaN values: {df['MAE'].isna().sum()}")
        print(f"StdDev NaN values: {df['StdDev'].isna().sum()}")

    # Plot-Manager initialisieren
    plots_dir = os.path.join(results_dir, "plots")
    plot_manager = PlotManager(plots_dir)

    # Plotter registrieren
    plot_manager.register_plotter('overview', ModelComparisonPlotter(plots_dir))
    plot_manager.register_plotter('heatmap', HeatmapPlotter(plots_dir))
    plot_manager.register_plotter('model_heatmap', ModelHeatmapPlotter(plots_dir))
    plot_manager.register_plotter('prediction_overview', PredictionPlotter(plots_dir))
    plot_manager.register_plotter('prediction_model', PredictionPlotterPerModel(plots_dir))
    plot_manager.register_plotter('prediction_dataset', PredictionPlotterPerDataset(plots_dir))
    plot_manager.register_plotter('geometry_mae', MAEPlotterGeometry(plots_dir))
    plot_manager.register_plotter('force_mae', MAEPlotterForce(plots_dir))
    plot_manager.register_plotter('mrr_mae', MAEPlotterMRR(plots_dir))
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
                   n_drop_values=20, patience=5, plot_types=None, use_phys_reference=False,
                   experiment_name = 'Experiment'):
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
                predictions.append(pred[n_drop_values:-n_drop_values].tolist())

            # Raw data als Dictionary speichern (JSON-serialisierbar)
            raw_data_dict = {
                'columns': raw_data[j].columns.tolist(),
                'data': raw_data[j].iloc[n_drop_values:-n_drop_values].to_dict('records')  # Als Liste von Dictionaries
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
                y_test[j].iloc[n_drop_values:-n_drop_values].values.tolist() if isinstance(y_test[j], pd.DataFrame) else y_test[j][
                                                                                                            n_drop_values:-n_drop_values].tolist(),
                raw_data_dict  # RawData als Dictionary
            ])
            header_list[j].append(name)

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
    if use_phys_reference:
        model = mphys.get_reference()
        reference_models.append(model)

    # Add datasets to meta information
    for data_params in dataSets:
        data_info = {
            **data_params.get_documentation()
        }
        meta_information["DataSets"].append(data_info)

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

        # Train and test reference models
        for idx, model in enumerate(reference_models):
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            for _ in range(NUMBEROFMODELS):
                model = reference_models_copy[idx]
                model.target_channel = dataClass.target_channels[0]
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
                reference_models[idx] = model

            # Fehlerberechnung
            calculate_and_store_results(model, dataClass, nn_preds, y_test, df_list_results, results, header_list,
                                        n_drop_values, raw_data)

        if len(reference_models) > 0:
            criterion = reference_models[0].criterion
        else:
            criterion = None
        # Train and test models
        for idx, model in enumerate(models):
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            for _ in range(NUMBEROFMODELS):
                model = models_copy[idx]
                if hasattr(model, 'input_size'):
                    model.input_size = None
                model.target_channel = dataClass.target_channels[0]
                model.train_model(X_train, y_train, X_val, y_val, n_epochs=NUMBEROFEPOCHS, patience=patience)
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
                models[idx] = model # save for documentation
            # Fehlerberechnung
            calculate_and_store_results(model, dataClass, nn_preds, y_test, df_list_results, results, header_list,
                                        n_drop_values, raw_data)

    #debug_results_structure(results)


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
    for model in models:
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

def reconstruct_results_dataframe(json_file_path):
    """
    Reconstruct the results DataFrame from the JSON file.

    Parameters:
    - json_file_path: Path to the JSON file containing the results.

    Returns:
    - df: Reconstructed DataFrame.
    """
    # Load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        documentation = json.load(json_file)

    # Extract the results from the JSON file
    results = documentation.get("Results", {}).get("Model_Comparison", [])

    # Define the columns for the DataFrame
    columns = HEADER

    # Create the DataFrame
    df = pd.DataFrame(results, columns=columns)

    return df

# Alternative: Hilfsfunktion für RawData-Rekonstruktion
def reconstruct_raw_data(raw_data_dict):
    """
    Hilfsfunktion zur Rekonstruktion von RawData aus verschiedenen Formaten.

    Parameters:
    - raw_data_dict: Dictionary mit RawData

    Returns:
    - pd.DataFrame: Rekonstruierte RawData oder None bei Fehler
    """
    if not isinstance(raw_data_dict, dict):
        return None

    try:
        if 'columns' in raw_data_dict and 'data' in raw_data_dict:
            # Format: {'columns': [...], 'data': [...]}
            return pd.DataFrame(raw_data_dict['data'], columns=raw_data_dict['columns'])
        else:
            # Direktes DataFrame-Dictionary Format
            return pd.DataFrame(raw_data_dict)
    except Exception as e:
        print(f"Error reconstructing RawData: {e}")
        return None