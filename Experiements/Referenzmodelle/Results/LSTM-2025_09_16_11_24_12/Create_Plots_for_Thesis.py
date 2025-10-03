import copy
import json
import os
import ast
import re

import shap
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
import seaborn as sns
from numpy.exceptions import AxisError
from numpy.f2py.auxfuncs import throw_error
from sklearn.metrics import mean_absolute_error

import Helper.handling_hyperopt as hyperopt
import Helper.handling_data as hdata
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
from matplotlib.colors import LinearSegmentedColormap

HEADER = ["DataSet", "DataPath", "Model", "MAE", "StdDev", "MAE_Ensemble", "Predictions", "GroundTruth", "RawData"]
SAMPLINGRATE = 50
AXIS = 'x'

class BasePlotter(ABC):
    """Abstrakte Basisklasse für verschiedene Plot-Typen"""

    def __init__(self, output_dir: str, known_material: str = 'S235JR', known_geometry: str = 'Plate'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.known_material = known_material
        self.known_geometry = known_geometry

        # KIT Farbpalette
        self.kit_red = "#D30015"
        self.kit_green = "#009682"
        self.kit_yellow = "#FFFF00"
        self.kit_orange = "#FFC000"
        self.kit_blue = "#0C537E"
        self.kit_dark_blue = "#002D4C"
        self.kit_magenta = "#A3107C"

        # Zusätzliche Farben für mehr Modelle
        self.colors = [self.kit_red, self.kit_orange, self.kit_magenta, self.kit_green, self.kit_yellow]

        # Benutzerdefinierte Farbpalette erstellen (grün=gut, gelb=mittel, rot=schlecht)
        self.custom_cmap = LinearSegmentedColormap.from_list(
            'kit_colors',
            [self.kit_green, self.kit_yellow, self.kit_red],
            N=256
        )

        # Benutzerdefinierte Farbpalette erstellen (grün=gut, gelb=mittel, rot=schlecht)
        self.custom_cmap = LinearSegmentedColormap.from_list(
            'kit_colors',
            [self.kit_green, self.kit_yellow, self.kit_red],
            N=256
        )

        # Dictionary für Modell-Schlüsselwörter und zugehörige Farben
        self.model_color_mapping = {
            "Random Forest": self.kit_red,
            "RNN": self.kit_yellow,
            "NN": self.kit_orange,
            "Erd": self.kit_magenta,
            "Friction": self.kit_magenta,
            # Weitere Schlüsselwörter und Farben können hier hinzugefügt werden
        }

    def get_model_color(self, model_name: str):
        """
        Liefert die Farbe für ein Modell basierend auf Schlüsselwörtern im Modellnamen.
        Standardmäßig wird None zurückgegeben, wenn kein Schlüsselwort gefunden wird.
        """
        for keyword, color in self.model_color_mapping.items():
            if keyword.lower() in model_name.lower():
                return color
        return None

    @abstractmethod
    def create_plots(self, df: pd.DataFrame, **kwargs):
        """Erstellt Plots basierend auf dem übergebenen DataFrame"""
        pass

    def save_plot(self, fig, filename: str):
        """Speichert einen Plot"""
        plot_path = os.path.join(self.output_dir, filename)
        fig.savefig(plot_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        return plot_path

    def parse_filename(self, filename: str):
        """Parst den Dateinamen und extrahiert Material und Geometrie"""
        # Entferne .csv Extension
        name_without_ext = filename.replace('.csv', '')

        # Entferne Maschinen Bezeichung
        name_without_ext = name_without_ext.replace('DMC_', '').replace('DMC60H_', '')
        # Split nach Unterstrichen
        parts = name_without_ext.split('_')

        #print(f"Debug: Parsing '{filename}' -> Parts: {parts}")

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

        #print(f"Debug: '{filename}' -> Material: '{material}', Geometry: '{geometry}'")
        return material, geometry

    def extract_material_and_geometry(self, df, datapath_column='DataPath'):
        """
        Extrahiere Material und Geometrie aus der DataPath-Spalte und füge sie als neue Spalten hinzu.

        Parameter:
        - df: DataFrame mit der Spalte 'DataPath'
        - datapath_column: Name der Spalte, die die DataPath-Informationen enthält (Standard: 'DataPath')

        Rückgabe:
        - DataFrame mit den zusätzlichen Spalten 'Material' und 'Geometry'
        """
        # Kopie des DataFrames, um das Original nicht zu verändern
        df = df.copy()

        # Extrahiere Material und Geometrie für jeden Eintrag in der DataPath-Spalte
        df[['Material', 'Geometry']] = df[datapath_column].apply(
            lambda x: pd.Series(self.parse_filename(x))
        )

        return df

    def replace_material_names(self, materials: list) -> list:
        materials = materials.copy()
        for idx, mat in enumerate(materials):
            materials[idx] = self.replace_material_name(mat)
        return materials

    def replace_material_name(self, material: str) -> str:
        material_mapping = {
            'S235JR': 'Stahl',
            'AL_2007_T4': 'Aluminium',
            'AL2007T4': 'Aluminium',
            # Weitere Zuordnungen hier
        }
        return material_mapping.get(material, material)

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

    def create_ordered_categories_from_datapath(self, dataset_paths: list):
        """Ordnet die Kategorien so, dass known oben links und unknown unten rechts sind"""
        materials, geometries = set(), set()

        # Material und Geometrie aus Dataset_Path extrahieren
        for path in dataset_paths:
            material, geometry = self.parse_filename(path)
            materials.add(material)
            geometries.add(geometry)

        # Verwende create_ordered_categories, um die sortierten Listen zu erhalten
        materials_ordered, geometries_ordered = self.create_ordered_categories(list(materials), list(geometries))

        # Dataset_Paths in gewünschter Reihenfolge erstellen
        ordered_datasets = []
        for material in materials_ordered:
            for geometry in geometries_ordered:
                dataset_name = f"{material}_{geometry}"
                if dataset_name in dataset_paths:
                    ordered_datasets.append(dataset_name)

        return ordered_datasets

    def get_model_seed_columns(self, df_columns: list, model_base_name: str):
        """Findet alle Seed-Spalten für ein bestimmtes Modell"""
        seed_columns = [col for col in df_columns if col.startswith(f'{model_base_name}')]
        if len(seed_columns) == 0:
            seed_columns = [col for col in df_columns if col == model_base_name]
        return seed_columns

    @staticmethod
    def replace_space(text):
        # Erstelle eine Kopie des Textes für die Verarbeitung
        ergebnis = text

        # Spezifische Ersetzung für 'Recurrent Neural Net'
        ergebnis = re.sub(r'Recurrent Neural Net', 'Recurrent\nNeural Net', ergebnis)

        # Spezifische Ersetzung für 'Random Forest'
        ergebnis = re.sub(r'Random Forest', 'Random\nForest', ergebnis)

        # Bestehende Ersetzung für \w+Sampler
        ergebnis = re.sub(r'(\w+Sampler)', r'\n\1', ergebnis)
        # Trenne die art des Samplers vom Wort Sampler und fügt einen umbruch hinzu
        ergebnis = re.sub(r'(\w+)(Sampler)', r'\1-\n\2', ergebnis)

        ergebnis = ergebnis.replace('Grid', 'Raster')
        ergebnis = ergebnis.replace('Random', 'Zufalls')

        return ergebnis

class HeatmapPlotter(BasePlotter):
    """Erstellt Heatmaps für den Vergleich mehrerer ML-Modelle mit MAE und Standardabweichung"""

    def format_cell_annotation(self, mae_value, std_value):
        """Formatiert die Zellenannotation mit MAE und Standardabweichung"""
        if pd.isna(mae_value) or pd.isna(std_value):
            return ""
        if std_value == 0 or std_value < 0.001:
            return f"{mae_value:.3f}"
        return f"{mae_value:.3f}\n±{std_value:.3f}"

    def get_text_color_for_background(self, mae_value):
        """Bestimmt die Textfarbe basierend auf dem Hintergrund (dunkel=weiß, hell=self.kit_dark_blue)"""
        if pd.isna(mae_value):
            return self.kit_dark_blue
        if 0.1 < mae_value < 0.2:
            return self.kit_dark_blue
        else:
            return 'white'

    def create_dataset_names(self, mae_df):
        """Erstellt kombinierte Dataset-Namen aus Material und Geometrie"""
        # Materialnamen ersetzen (z.B. für bessere Lesbarkeit)
        #mae_df['Material'] = self.replace_material_names(mae_df['Material'])
        # Dataset-Namen kombinieren
        mae_df['Dataset'] = mae_df['Material'] + '_' + mae_df['Geometry']
        return mae_df

    def create_plots(self, df: pd.DataFrame = None, title: str = 'Model Vergleich', model_names=None, **kwargs):
        """
        Erstellt eine Vergleichs-Heatmap für alle Modelle mit MAE und Standardabweichung
        Berücksichtigt jetzt auch verschiedene Trainingsdatasets (DataSet)

        Args:
            df: DataFrame mit den Daten
            title: Titel der Heatmap
            model_names: Dictionary für benutzerdefinierte Modellnamen
            **kwargs: Weitere Parameter
        """
        # MAE-Daten erstellen
        mae_df = self.extract_material_and_geometry(df)
        # Dataset-Namen erstellen
        mae_df = self.create_dataset_names(mae_df)

        # Unique Materialien, Geometrien, Modelle und TrainingsDatasets ermitteln
        materials = mae_df['Material'].unique()
        geometries = mae_df['Geometry'].unique()
        models = mae_df['Model'].unique()
        train_datasets = mae_df['DataSet'].unique()

        # Kategorien ordnen
        materials_ordered, geometries_ordered = self.create_ordered_categories(materials, geometries)

        # Test-Datasets in gewünschter Reihenfolge erstellen (für y-Achse)
        ordered_test_datasets = []
        for material in materials_ordered:
            for geometry in geometries_ordered:
                dataset_name = f"{material}_{geometry}"
                if dataset_name in mae_df['Dataset'].unique():
                    ordered_test_datasets.append(dataset_name)

        # Modell + TrainingsDataSet Kombinationen erstellen (für x-Achse)
        model_dataset_combinations = []
        model_dataset_labels = {}

        for model in sorted(models):  # Modelle sortieren für konsistente Reihenfolge
            for train_dataset in sorted(train_datasets):  # TrainingsDatasets sortieren
                combination = f"{model}|{train_dataset}"
                model_dataset_combinations.append(combination)

                # Labels für bessere Lesbarkeit erstellen
                clean_model = model.replace('Plate_TrainVal_', '').replace('Reference_', '').replace('ST_Data_', '') \
                    .replace('ST_Plate_Notch_', '').replace('Ref', '').replace('_', ' ')
                clean_dataset = train_dataset.replace('_', ' ')
                if len(train_datasets) == 1:
                    model_dataset_labels[combination] = f"{clean_model}"
                else:
                    model_dataset_labels[combination] = f"{clean_model}\n({clean_dataset})"
                model_dataset_labels[combination] = self.replace_space(model_dataset_labels[combination])

        # Neue Spalte für Kombination erstellen
        mae_df['Model_DataSet'] = mae_df['Model'] + '|' + mae_df['DataSet']

        # Pivot-Tabellen für MAE und STD erstellen (ohne Aggregation, da jede Kombination einzigartig sein sollte)
        mae_pivot = mae_df.pivot_table(
            values='MAE',
            index='Dataset',
            columns='Model_DataSet',
            aggfunc='first'  # 'first' statt 'mean', da jede Kombination eindeutig sein sollte
        )
        std_pivot = mae_df.pivot_table(
            values='StdDev',
            index='Dataset',
            columns='Model_DataSet',
            aggfunc='first'
        )

        # Reorder basierend auf der gewünschten Reihenfolge
        mae_pivot = mae_pivot.reindex(index=ordered_test_datasets)
        std_pivot = std_pivot.reindex(index=ordered_test_datasets)

        # Spalten in der gewünschten Reihenfolge sortieren
        available_combinations = [combo for combo in model_dataset_combinations if combo in mae_pivot.columns]
        mae_pivot = mae_pivot.reindex(columns=available_combinations)
        std_pivot = std_pivot.reindex(columns=available_combinations)

        # Spalten umbenennen für bessere Lesbarkeit
        if model_names is None:
            mae_pivot.columns = [model_dataset_labels.get(col, col) for col in mae_pivot.columns]
            std_pivot.columns = [model_dataset_labels.get(col, col) for col in std_pivot.columns]
        else:
            # Bei benutzerdefinierten Namen: Modellnamen anwenden und TrainingsDataSet anhängen
            new_labels = []
            for col in mae_pivot.columns:
                model_part, dataset_part = col.split('|')
                model_name = model_names.get(model_part, model_part)
                clean_dataset = dataset_part.replace('_', ' ')
                new_labels.append(f"{model_name}\n({clean_dataset})")
            mae_pivot.columns = new_labels
            std_pivot.columns = new_labels

        # Annotations-Matrix und Textfarben-Matrix erstellen
        annotations = np.empty(mae_pivot.shape, dtype=object)
        text_colors = np.empty(mae_pivot.shape, dtype=object)

        for i in range(mae_pivot.shape[0]):
            for j in range(mae_pivot.shape[1]):
                mae_val = mae_pivot.iloc[i, j]
                std_val = std_pivot.iloc[i, j]
                annotations[i, j] = self.format_cell_annotation(mae_val, std_val)
                text_colors[i, j] = self.get_text_color_for_background(mae_val)

        # Dynamische Figurengröße basierend auf Anzahl der Kombinationen
        n_combinations = len(mae_pivot.columns)
        fig_width = max(14, n_combinations * 1.5)  # Mindestens 14, aber skaliert mit Anzahl Kombinationen
        fig_height = max(12, len(mae_pivot.index) * 0.8)  # Skaliert mit Anzahl Test-Datasets

        # Heatmap erstellen
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.set_dpi(1200)

        # Maske für NaN-Werte
        mask = mae_pivot.isna()

        # Schriftgrößen anpassen basierend auf Anzahl der Kombinationen
        titlesize = 40
        maesize = max(20, min(35, 200 // n_combinations))  # Dynamische Anpassung
        textsize = max(5, min(25, 100 // (n_combinations)))
        labelsize = 35

        # Heatmap mit seaborn für bessere Optik
        sns.heatmap(
            mae_pivot,
            annot=annotations,
            fmt='',
            cmap=self.custom_cmap,
            mask=mask,
            cbar_kws={'label': 'MAE'},
            ax=ax,
            square=False,
            vmin=0.04,
            vmax=0.31,
            linewidths=0.5,  # Leichte Linien zur besseren Abgrenzung
            linecolor='white',
            annot_kws={'size': maesize, 'weight': 'bold', 'ha': 'center', 'va': 'center'}
        )

        # Textfarben für jede Zelle individuell setzen
        for i in range(mae_pivot.shape[0]):
            for j in range(mae_pivot.shape[1]):
                if not pd.isna(mae_pivot.iloc[i, j]):
                    text = ax.texts[i * mae_pivot.shape[1] + j]
                    text.set_color(text_colors[i, j])

        # Titel und Labels mit größerer Schrift
        ax.set_title(f'MAE Heatmap:\nLSTM', fontsize=titlesize, fontweight='bold', pad=20, color=self.kit_dark_blue)
        ax.set_xlabel('Modell', fontsize=labelsize, fontweight='bold', color=self.kit_dark_blue)
        ax.set_ylabel('Test-Datensatz', fontsize=labelsize, fontweight='bold',
                      color=self.kit_dark_blue)

        # Achsenbeschriftungen vergrößern und Farbe setzen
        ax.tick_params(axis='both', which='major', labelsize=textsize, colors=self.kit_dark_blue)

        # Y-Achsen-Tick-Labels anpassen (Test-Datasets)
        y_labels = []
        for dataset in ordered_test_datasets:
            material, geometry = self.parse_filename(dataset)
            # Materialnamen ersetzen (z. B. S235JR -> Stahl)
            material = self.replace_material_name(material)
            y_labels.append(f"{material}\n{geometry}")

        ax.set_yticklabels(y_labels, fontsize=textsize, color=self.kit_dark_blue, ha='center', va='center')

        # X-Achsen-Tick-Labels sind bereits durch die Spaltenumbenennung gesetzt
        # Labels rotieren für bessere Lesbarkeit bei vielen Kombinationen
        rotation_angle = 45 if n_combinations > 8 else 0
        ax.set_xticklabels(mae_pivot.columns, fontsize=textsize, color=self.kit_dark_blue,
                           ha='right' if rotation_angle > 0 else 'center', rotation=rotation_angle)

        ax.tick_params(axis='x', pad=15)  # Abstand für x-Achse
        ax.tick_params(axis='y', pad=35)  # Abstand für y-Achse

        # Achsenbeschriftungen (Tick-Labels) explizit färben
        for label in ax.get_xticklabels():
            label.set_color(self.kit_dark_blue)
        for label in ax.get_yticklabels():
            label.set_color(self.kit_dark_blue)

        # Colorbar-Label vergrößern
        cbar = ax.collections[0].colorbar
        cbar.set_label('MAE $I$ in A', fontsize=labelsize, fontweight='bold', color=self.kit_dark_blue)
        cbar.ax.tick_params(labelsize=labelsize, colors=self.kit_dark_blue)

        # Colorbar Tick-Labels explizit färben
        for label in cbar.ax.get_yticklabels():
            label.set_color(self.kit_dark_blue)

        plt.tight_layout()

        # Speichern mit der save_plot Methode der Basisklasse
        filename = f'heatmap_LSTM.png'
        plot_path = self.save_plot(fig, filename)
        print(f"Erweiterte Model Comparison Heatmap mit TrainingsDatasets und Standardabweichung erstellt: {plot_path}")
        return [plot_path]

class ModelHeatmapPlotter(BasePlotter):
    """Erstellt separate Heatmaps für jedes ML-Modell mit MAE und Standardabweichung"""

    def format_cell_annotation(self, mae_value, std_value):
        """Formatiert die Zellenannotation mit MAE und Standardabweichung"""
        if pd.isna(mae_value) or pd.isna(std_value):
            return ""
        if std_value == 0 or std_value < 0.001:
            return f"{mae_value:.3f}"
        return f"{mae_value:.3f}\n±{std_value:.3f}"

    def get_text_color_for_background(self, mae_value):
        """Bestimmt die Textfarbe basierend auf dem Hintergrund (dunkel=weiß, hell=self.kit_dark_blue)"""
        # Verwende die KIT-Farben aus der Basisklasse
        kit_dark_blue = "#002D4C"

        if pd.isna(mae_value):
            return kit_dark_blue

        # Bei Werten < 0.5 (dunkler Hintergrund) verwende kit_dark_blue, sonst weiß
        if 0.1 < mae_value < 0.2:
            return kit_dark_blue
        else:
            return 'white'

    def create_plots(self, df: pd.DataFrame = None, **kwargs):
        """
        Erstellt für jedes Modell eine separate Heatmap mit MAE und Standardabweichung

        Args:
            df: DataFrame
            **kwargs: Weitere Parameter
        """
        # KIT-Farben aus der Basisklasse verwenden
        kit_dark_blue = "#002D4C"

        # MAE-Daten erstellen
        mae_df = self.extract_material_and_geometry(df)

        # Unique Materialien und Geometrien ermitteln
        materials = mae_df['Material'].unique()
        geometries = mae_df['Geometry'].unique()
        models = mae_df['Model'].unique()

        # Kategorien ordnen (ohne (Bekannt)/(Unbekannt) in den Pivot-Tabellen)
        materials_ordered, geometries_ordered = self.create_ordered_categories(materials, geometries)
        plot_paths = []

        # Für jedes Modell eine separate Heatmap erstellen
        for model in models:
            model_data = mae_df[mae_df['Model'] == model]
            # Pivot-Tabellen für MAE und STD erstellen
            mae_pivot = model_data.pivot_table(
                values='MAE',
                index='Material',
                columns='Geometry',
                aggfunc='mean'
            )
            std_pivot = model_data.pivot_table(
                values='StdDev',
                index='Material',
                columns='Geometry',
                aggfunc='mean'
            )
            # Reorder basierend auf der gewünschten Reihenfolge
            mae_pivot = mae_pivot.reindex(index=materials_ordered, columns=geometries_ordered)
            std_pivot = std_pivot.reindex(index=materials_ordered, columns=geometries_ordered)
            # Annotations-Matrix und Textfarben-Matrix erstellen
            annotations = np.empty(mae_pivot.shape, dtype=object)
            text_colors = np.empty(mae_pivot.shape, dtype=object)
            for i in range(mae_pivot.shape[0]):
                for j in range(mae_pivot.shape[1]):
                    mae_val = mae_pivot.iloc[i, j]
                    std_val = std_pivot.iloc[i, j]
                    annotations[i, j] = self.format_cell_annotation(mae_val, std_val)
                    text_colors[i, j] = self.get_text_color_for_background(mae_val)

            # Heatmap erstellen
            fig, ax = plt.subplots(figsize=(14, 12))
            fig.set_dpi(1200)
            # Maske für NaN-Werte
            mask = mae_pivot.isna()
            titlesize = 40
            maesize = 45
            textsize = 35
            labelsize = 35

            # Heatmap mit seaborn für bessere Optik
            sns.heatmap(
                mae_pivot,
                annot=annotations,
                fmt='',
                cmap=self.custom_cmap,
                mask=mask,
                cbar_kws={'label': 'MAE'},
                ax=ax,
                square=True,
                vmin=0.04,
                vmax=0.31,
                linewidths=0.0,
                linecolor='white',
                annot_kws={'size': maesize, 'weight': 'bold', 'ha': 'center', 'va': 'center'}
            )

            # Textfarben für jede Zelle individuell setzen
            for i in range(mae_pivot.shape[0]):
                for j in range(mae_pivot.shape[1]):
                    if not pd.isna(mae_pivot.iloc[i, j]):
                        text = ax.texts[i * mae_pivot.shape[1] + j]
                        text.set_color(text_colors[i, j])

            # Model-Name aufräumen
            model_clean = model.replace('Plate_TrainVal_', '').replace('Reference_', '').replace('ST_Data_',
                                                                                                 '').replace(
                'ST_Plate_Notch_', '').replace('Ref', '').replace('_', ' ')

            # Titel und Labels mit größerer Schrift
            title = re.sub(r'([a-zA-Z]+)Sampler', r'\n\1Sampler', model_clean)
            ax.set_title(f'MAE Heatmap: {title}', fontsize=titlesize, fontweight='bold', pad=20,
                         color=self.kit_dark_blue)
            ax.set_xlabel('Geometry', fontsize=labelsize, fontweight='bold', color=self.kit_dark_blue)
            ax.set_ylabel('Material', fontsize=labelsize, fontweight='bold', color=self.kit_dark_blue)

            materials_ordered_renamed = self.replace_material_names(materials_ordered)
            # Achsenbeschriftungen anpassen: (Bekannt)/(Unbekannt) hinzufügen
            x_labels = [f"{geo}\n(Bekannt)" if geo == self.known_geometry else f"{geo}\n(Unbekannt)" for geo in
                        geometries_ordered]
            y_labels = [f"{mat}\n(Bekannt)" if mat == self.replace_material_name(self.known_material) else f"{mat}\n(Unbekannt)" for mat in
                        materials_ordered_renamed]

            # Tick-Labels setzen und zentrieren
            ax.set_xticklabels(
                x_labels,
                fontsize=textsize,
                color=self.kit_dark_blue,
                ha='center',  # Horizontal zentrieren
                va='center',  # Vertikal zentrieren
            )
            ax.set_yticklabels(
                y_labels,
                fontsize=textsize,
                color=self.kit_dark_blue,
                ha='center',  # Horizontal zentrieren
                va='center',  # Vertikal zentrieren
            )
            ax.tick_params(axis='x', pad=35)  # Abstand für x-Achse
            ax.tick_params(axis='y', pad=35)  # Abstand für y-Achse

            # Achsenbeschriftungen (Tick-Labels) explizit färben
            for label in ax.get_xticklabels():
                label.set_color(kit_dark_blue)
            for label in ax.get_yticklabels():
                label.set_color(kit_dark_blue)

            # Colorbar-Label vergrößern
            cbar = ax.collections[0].colorbar
            cbar.set_label('MAE $I$ in A', fontsize=textsize, fontweight='bold', color=self.kit_dark_blue)
            cbar.ax.tick_params(labelsize=labelsize, colors=self.kit_dark_blue)
            # Colorbar Tick-Labels explizit färben
            for label in cbar.ax.get_yticklabels():
                label.set_color(kit_dark_blue)

            plt.tight_layout()

            # Speichern mit der save_plot Methode der Basisklasse
            filename = f'heatmap_{model_clean.replace(" ", "_")}_with_std.png'
            plot_path = self.save_plot(fig, filename)
            plot_paths.append(plot_path)
            print(f"Heatmap mit Standardabweichung für {model_clean} erstellt: {plot_path}")

        return plot_paths

def calculate_mae_and_std(df, datapath, ground_truth_column='GroundTruth', model_prefixes=None):
    """
    Berechnet den MAE und die Standardabweichung für alle Modelle mit den angegebenen Präfixen.
    Extrahiere Material und Geometrie aus dem DataPath.

    Args:
        df: DataFrame mit den Vorhersagen und Ground Truth.
        datapath: Pfad zu den daten. Muss Namenskonvention erfüllen. Maschine_Material_Geometrie_Variante_Version.
        ground_truth_column: Name der Spalte mit den Ground Truth-Werten.
        model_prefixes: Liste der Modell-Präfixe, für die MAE und StdDev berechnet werden sollen.

    Returns:
        DataFrame mit den berechneten MAE- und StdDev-Werten für jedes Modell, inkl. Material und Geometrie.
    """
    if model_prefixes is None:
        model_prefixes = [
            'Reference_LSTM_RandomSampler',
            'Reference_LSTM_GridSampler',
            'Reference_LSTM_TPESampler'
        ]

    results = []
    plotter = HeatmapPlotter(output_dir='Plots_Thesis')  # Temporäres Objekt zur Nutzung von parse_filename

    for prefix in model_prefixes:
        # Alle Spalten mit dem aktuellen Präfix finden
        model_columns = [col for col in df.columns if col.startswith(prefix)]

        if not model_columns:
            continue

        # MAE für jeden Seed berechnen
        mae_values = []
        for col in model_columns:
            mae = mean_absolute_error(df[ground_truth_column], df[col])
            mae_values.append(mae)

        # Durchschnittlichen MAE und Standardabweichung berechnen
        avg_mae = np.mean(mae_values)
        std_mae = np.std(mae_values)

        # Material und Geometrie aus dem DataPath extrahieren
        material, geometry = plotter.parse_filename(datapath)

        # Ergebnisse speichern
        results.append({
            'Model': prefix,
            'MAE': avg_mae,
            'StdDev': std_mae,
            'DataSet': df['DataSet'].iloc[0] if 'DataSet' in df.columns else 'Unknown',
            'DataPath': datapath,
            'Material': material,
            'Geometry': geometry
        })

    return pd.DataFrame(results)

if __name__ == '__main__':
    # Liste für alle DataFrames mit MAE/StdDev
    all_mae_std_dfs = []

    for file in os.listdir('Predictions'):
        if file.endswith('.csv'):
            df = pd.read_csv(f"Predictions/{file}")
            # MAE und StdDev für die Sampler berechnen
            mae_std_df = calculate_mae_and_std(df, file)
            all_mae_std_dfs.append(mae_std_df)

    # Alle DataFrames zu einem einzigen DataFrame kombinieren
    combined_mae_std_df = pd.concat(all_mae_std_dfs, ignore_index=True)

    # HeatmapPlotter initialisieren
    plotter = HeatmapPlotter(output_dir='Plots_Thesis')

    # Heatmap erstellen
    plot_paths = plotter.create_plots(
        df=combined_mae_std_df,
        title='Ergebnisse der\nHyperparameteroptimierung'
    )

    print(f"Heatmaps wurden erstellt: {plot_paths}")

    # ModelHeatmapPlotter initialisieren
    plotter = ModelHeatmapPlotter(output_dir='Plots_Thesis')

    # Heatmaps für jedes Modell erstellen
    plot_paths = plotter.create_plots(df=combined_mae_std_df)

    print(f"Heatmaps wurden erstellt: {plot_paths}")
