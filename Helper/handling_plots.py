import os
import re

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from typing import Dict, List, Optional

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

        self.Metrik = 'nMAE'

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
        self.color_vmin = 0.04
        self.color_vmax = 0.41

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
        name = filename.replace('\n', '_')
        plot_path = os.path.join(self.output_dir, name)
        fig.savefig(plot_path + '.svg', dpi=600, bbox_inches='tight')
        fig.savefig(plot_path + '.pdf', dpi=600, bbox_inches='tight')
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

        # Bestehende Ersetzung für \w+Sampler
        ergebnis = re.sub(r'(\w+Sampler)', r'\n\1', ergebnis)
        # Trenne die art des Samplers vom Wort Sampler und fügt einen umbruch hinzu
        ergebnis = re.sub(r'(\w+)(Sampler)', r'\1-\n\2', ergebnis)
        ergebnis = ergebnis.replace('Grid', 'Raster')
        ergebnis = ergebnis.replace('Random', 'Zufalls')

        # Spezifische Ersetzung für 'Random Forest'
        ergebnis = re.sub(r'Zufalls Forest', 'Random\nForest', ergebnis)

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
        if 0.15 < mae_value < 0.3:
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

    def create_plots(self, df: pd.DataFrame = None, title: str = 'Model Vergleich', filename_postfix = 'Vergleich', model_names=None, new_names = None, **kwargs):
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
                # Model-Name aufräumen
                if new_names is not None:
                    model_clean = new_names[model]
                else:
                    model_clean = model.replace('Plate_TrainVal_', '').replace('Reference_', '').replace('ST_Data_',
                                                                                                         '').replace(
                        'ST_Plate_Notch_', '').replace('Ref', '').replace('_', ' ')

                clean_dataset = train_dataset.replace('_', ' ')

                if len(train_datasets) == 1:
                    model_dataset_labels[combination] = f"{model_clean}"
                else:
                    model_dataset_labels[combination] = f"{model_clean}\n({clean_dataset})"
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
        maesize = max(20, min(35, 150 // n_combinations))  # Dynamische Anpassung
        textsize = max(5, min(25, 200 // n_combinations))
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
            vmin=self.color_vmin,
            vmax=self.color_vmax,
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
        ax.set_title(f'{self.Metrik} Heatmap:{title}', fontsize=titlesize, fontweight='bold', pad=20, color=self.kit_dark_blue)
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
        cbar.set_label(self.Metrik, fontsize=labelsize, fontweight='bold', color=self.kit_dark_blue)
        cbar.ax.tick_params(labelsize=labelsize, colors=self.kit_dark_blue)

        # Colorbar Tick-Labels explizit färben
        for label in cbar.ax.get_yticklabels():
            label.set_color(self.kit_dark_blue)

        plt.tight_layout()

        # Speichern mit der save_plot Methode der Basisklasse
        filename = f'heatmap_{filename_postfix}'
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
        if 0.15 < mae_value < 0.3:
            return kit_dark_blue
        else:
            return 'white'

    def create_plots(self, df: pd.DataFrame = None, new_names= None, **kwargs):
        """
        Erstellt für jedes Modell eine separate Heatmap mit MAE und Standardabweichung

        Args:
            df: DataFrame
            new_names: Anzeige Namen der modelle
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
                vmin=self.color_vmin,
                vmax=self.color_vmax,
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
            if new_names is not None:
                model_clean = new_names[model]
            else:
                model_clean = model.replace('Plate_TrainVal_', '').replace('Reference_', '').replace('ST_Data_',
                                                                                                     '').replace(
                    'ST_Plate_Notch_', '').replace('Ref', '').replace('_', ' ')

            # Titel und Labels mit größerer Schrift
            title = re.sub(r'([a-zA-Z]+)Sampler', r'\n\1Sampler', model_clean)
            #title = title.replace('Recurrent Neural Net', 'Rekurrentes neuronales Netz')
            ax.set_title(f'{self.Metrik} Heatmap: {title}', fontsize=titlesize, fontweight='bold', pad=20,
                         color=self.kit_dark_blue)
            ax.set_xlabel('Geometrie', fontsize=labelsize, fontweight='bold', color=self.kit_dark_blue)
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
            cbar.set_label(self.Metrik, fontsize=textsize, fontweight='bold', color=self.kit_dark_blue)
            cbar.ax.tick_params(labelsize=labelsize, colors=self.kit_dark_blue)
            # Colorbar Tick-Labels explizit färben
            for label in cbar.ax.get_yticklabels():
                label.set_color(kit_dark_blue)

            plt.tight_layout()

            # Speichern mit der save_plot Methode der Basisklasse
            filename = f'heatmap_{model_clean.replace(" ", "_")}_with_std'
            plot_path = self.save_plot(fig, filename)
            plot_paths.append(plot_path)
            print(f"Heatmap mit Standardabweichung für {model_clean} erstellt: {plot_path}")

        return plot_paths

def calculate_nmae_and_std(df, datapath, model_prefixes, ground_truth_column='GroundTruth'):
    """
    Berechnet den nRMSE und die Standardabweichung für alle Modelle mit den angegebenen Präfixen.
    Extrahiere Material und Geometrie aus dem DataPath.

    Args:
        df: DataFrame mit den Vorhersagen und Ground Truth.
        datapath: Pfad zu den daten. Muss Namenskonvention erfüllen. Maschine_Material_Geometrie_Variante_Version.
        model_prefixes: Liste der Modell-Präfixe, für die MAE und StdDev berechnet werden sollen.
        ground_truth_column: Name der Spalte mit den Ground Truth-Werten.

    Returns:
        DataFrame mit den berechneten nRMSE- und StdDev-Werten für jedes Modell, inkl. Material und Geometrie.
    """

    results = []
    plotter = HeatmapPlotter(output_dir='Plots_Thesis')  # Temporäres Objekt zur Nutzung von parse_filename

    for prefix in model_prefixes:
        # Alle Spalten mit dem aktuellen Präfix finden
        model_columns = [col for col in df.columns if col.startswith(prefix)]

        if not model_columns:
            continue

        # MAE für jeden Seed berechnen
        rf_mse_values = []
        for col in model_columns:
            # RMSE berechnen
            mae =  mean_absolute_error(df[ground_truth_column], df[col].fillna(0))

            # Mittelwert der tatsächlichen Werte
            mean_ground_truth = df[ground_truth_column].abs().mean()

            # CV(RMSE) berechnen
            nmae = mae / mean_ground_truth
            rf_mse_values.append(nmae)

        # Durchschnittlichen MAE und Standardabweichung berechnen
        avg_mae = np.mean(rf_mse_values)
        std_mae = np.std(rf_mse_values)

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

def plot_time_series(
    data: pd.DataFrame,
    title: str,
    filename: str,
    dpi: int = 300,
    col_name: str = 'curr_x',
    label: str = '$I$\nin A',
    y_configs: List[Dict[str, str]] = None,
    f_a: int = 50,
    path: str = 'Plots',
) -> None:
    """
    Erstellt einen DIN 461 konformen Zeitverlaufsplan mit:
    - Plot: Stromvorhersage mit farblicher Kennzeichnung von Bereichen mit |v_x| < 1 m/s

    Args:
        data: DataFrame mit den Daten
        title: Titel des Plots
        filename: Dateiname zum Speichern des Plots
        dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
        col_name: Spaltenname für Strom-Messwerte
        label: Beschriftung der y-Achse für Strom
        y_configs: Liste von Dictionaries mit 'ycolname' und 'ylabel' für zusätzliche y-Achsen
                  Beispiel: [{'ycolname': 'Abweichung_RF', 'ylabel': 'Random Forest'},
                             {'ycolname': 'Abweichung_RNN', 'ylabel': 'RNN'}]
        f_a: Abtastfrequenz in Hz (Standard: 50)
        path: Pfad zum Speichern der Plots
    """
    fontsize_axis = 14
    fontsize_axis_label = 16
    fontsize_title = 18
    line_size = 1.5

    # KIT-Farben
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_orange = "#FFC000"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_magenta = "#A3107C"
    kit_gray = "#767676"

    # Standardfarben für zusätzliche y-Achsen (kann erweitert werden)
    y_colors = [kit_red, kit_orange, kit_magenta, kit_yellow, kit_green]

    time = data.index / f_a

    # Erstelle Figure
    fig, ax_i = plt.subplots(figsize=(12, 8), dpi=dpi)

    # ----- Unterer Plot (Stromvorhersage) -----
    ax_i.spines['left'].set_position('zero')
    ax_i.spines['bottom'].set_position('zero')
    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)
    ax_i.spines['left'].set_color(kit_dark_blue)
    ax_i.spines['bottom'].set_color(kit_dark_blue)
    ax_i.spines['left'].set_linewidth(line_size)
    ax_i.spines['bottom'].set_linewidth(line_size)

    # Plot für Vorhersagen (dynamisch basierend auf y_configs)
    lines_pred = []
    if y_configs is None:
        y_configs = []

    for i, config in enumerate(y_configs):
        color = y_colors[i % len(y_colors)]
        line, _ = plot_prediction_with_std(
            data,
            config['ycolname'],
            color,
            config['ylabel']
        )
        if line is not None:
            lines_pred.append(line)

    # Plot für Strom-Messwerte
    line_i, = ax_i.plot(time, data[col_name], label='Strom-Messwerte',
                       color=kit_blue, linewidth=2)

    # Grid und Achsenbeschriftung
    ax_i.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=line_size/2)
    ax_i.set_axisbelow(True)
    ax_i.tick_params(axis='both', colors=kit_dark_blue, labelsize=fontsize_axis_label)

    # X-Achsenbeschriftung (nur beim unteren Plot)
    xmin, xmax = ax_i.get_xlim()
    ymin, ymax = ax_i.get_ylim()
    x_pos = -0.08 * (xmax - xmin)
    y_pos = -0.15 * ymax
    arrow_length = 0.04 * (xmax - xmin)

    ax_i.set_xlim(left=min(x_pos, xmin), right=xmax + arrow_length/2) # Achsenbegrenzungen anpassen

    ax_i.annotate('', xy=(xmax + arrow_length/2, 0), xytext=(xmax - arrow_length/2, 0),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_i.text(xmax*0.95, y_pos, r'$t$ in s',
             ha='left', va='center', color=kit_dark_blue, fontsize=fontsize_axis)

    # Y-Achsenbeschriftung
    y_pos = ymax * 0.85
    arrow_length = 0.04*(ymax-ymin)

    ax_i.set_ylim(bottom=min(y_pos, ymin), top=ymax + arrow_length / 2) # Achsenbegrenzungen anpassen

    ax_i.annotate('', xy=(0, ymax + arrow_length/2),
                 xytext=(0, ymax - arrow_length/2),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_i.text(x_pos, y_pos - 0.04*(ymax-ymin), label,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=fontsize_axis)

    # Titel über beiden Plots
    fig.suptitle(title, color=kit_dark_blue, fontsize=fontsize_title,
                fontweight='bold', y=0.98)

    '''    
    # Kombinierte Legende
    legend_elements = [line_i] + lines_pred
    legend_labels = [line.get_label() for line in legend_elements if line is not None]

    fig.legend(
        handles=legend_elements,
        labels=legend_labels,
        loc='lower center',
        ncol=3,
        frameon=True,
        facecolor='white',
        edgecolor=kit_dark_blue,
        framealpha=1.0,
        bbox_to_anchor=(0.5, -0.05)
    )'''

    # Legende (inkl. Näherungslinien)
    legend_elements = [line_i] + lines_pred
    legend_labels = [line.get_label() for line in legend_elements]

    if len(legend_elements) > 1:
        fig.legend(
            handles=legend_elements,
            labels=legend_labels,
            loc='lower center',
            ncol=2,  # 4 Spalten für bessere Lesbarkeit
            frameon=True,
            facecolor='white',
            edgecolor=kit_dark_blue,
            framealpha=1.0,
            fontsize = fontsize_axis_label,
        )

    '''    
    # Kombinierte Legende für die Achsen
    legend_elements = [line_i] + lines_pred
    legend_labels = [line.get_label() for line in legend_elements if line is not None]
    lines = [line for line in legend_elements if line is not None]
    labels = [line.get_label() for line in lines if line is not None]
    legend = ax_i.legend(lines, legend_labels, loc='upper right',
                        frameon=True, fancybox=False, shadow=False,
                        framealpha=1.0, facecolor='white', edgecolor=kit_dark_blue)
    legend.get_frame().set_linewidth(1.0)
    for text in legend.get_texts():
        text.set_color(kit_dark_blue)
    '''

    # Speichern des Plots
    plot_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    fig.savefig(plot_path + '.svg', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(plot_path + '.pdf', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved as {plot_path}')

def plot_time_series_with_sections(
    data: pd.DataFrame,
    title: str,
    filename: str,
    dpi: int = 300,
    col_name: str = 'curr_x',
    label: str = '$I$\nin A',
    y_configs: List[Dict[str, str]] = None,
    f_a: int = 50,
    path: str = 'Plots',
    v_colname: str = 'v_x',
    v_label: str = 'Vorschubgeschwindigkeit',
    v_axis: str = 'v in m/s',
    v_threshold: float = 1.0,
) -> None:
    """
    Erstellt einen DIN 461 konformen Zeitverlaufsplan mit:
    - Plot: Stromvorhersage mit farblicher Kennzeichnung von Bereichen mit |v_x| < 1 m/s

    Args:
        data: DataFrame mit den Daten
        title: Titel des Plots
        filename: Dateiname zum Speichern des Plots
        dpi: Auflösung des Plots in Dots Per Inch (Standard: 300)
        col_name: Spaltenname für Strom-Messwerte
        label: Beschriftung der y-Achse für Strom
                v_colname: Spaltenname für Vorschubgeschwindigkeit
        v_label: Legendenlabel für Vorschubgeschwindigkeit
        v_axis: Beschriftung der y-Achse für Vorschub
        v_threshold: Grenzwert für die gekennzeichneten Bereiche.
        y_configs: Liste von Dictionaries mit 'ycolname' und 'ylabel' für zusätzliche y-Achsen
                  Beispiel: [{'ycolname': 'Abweichung_RF', 'ylabel': 'Random Forest'},
                             {'ycolname': 'Abweichung_RNN', 'ylabel': 'RNN'}]
        f_a: Abtastfrequenz in Hz (Standard: 50)
        path: Pfad zum Speichern der Plots
    """
    fontsize_axis = 14
    fontsize_axis_label = 16
    fontsize_title = 18
    line_size = 1.5
    plot_line_size = 2

    # KIT-Farben
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_orange = "#FFC000"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_magenta = "#A3107C"
    kit_gray = "#767676"

    # Standardfarben für zusätzliche y-Achsen (kann erweitert werden)
    y_colors = [kit_red, kit_orange, kit_magenta, kit_yellow, kit_green]

    time = data.index / f_a

    # Erstelle Figure mit zwei Achsen
    fig, (ax_v, ax_i) = plt.subplots(
        2, 1, figsize=(12, 10), dpi=dpi,
        sharex=True, height_ratios=[1, 3],
        gridspec_kw={'hspace': 0.05}
    )

    # ----- Oberer Plot (Vorschubgeschwindigkeit) -----
    ax_v.spines['left'].set_position('zero')
    ax_v.spines['bottom'].set_position('zero')
    ax_v.spines['top'].set_visible(False)
    ax_v.spines['right'].set_visible(False)
    ax_v.spines['left'].set_color(kit_dark_blue)
    ax_v.spines['bottom'].set_color(kit_dark_blue)
    ax_v.spines['left'].set_linewidth(line_size)
    ax_v.spines['bottom'].set_linewidth(line_size)

    # Plot Vorschubgeschwindigkeit
    line_v, = ax_v.plot(time, data[v_colname], label=v_label,
                       color=kit_blue, linewidth=plot_line_size)

    # Grid und Achsenbeschriftung
    ax_v.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5)
    ax_v.set_axisbelow(True)
    ax_v.tick_params(axis='both', colors=kit_dark_blue, labelsize=fontsize_axis_label)

    # Achsenbeschriftung
    xmin, xmax = ax_v.get_xlim()
    ymin, ymax = ax_v.get_ylim()
    y_pos = -0.2 * ymax

    # X-Achsenbeschriftung
    ax_v.annotate('', xy=(xmax, 0), xytext=(xmax*0.95, 0),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_v.text(xmax*0.95, y_pos, r'$t$ in s',
             ha='left', va='center', color=kit_dark_blue, fontsize=fontsize_axis_label)

    # Y-Achsenbeschriftung
    x_label_pos_y = -0.06 * (xmax - xmin)
    y_label_pos_y = ymax * 0.6
    ax_v.annotate('', xy=(0, ymax),
                 xytext=(0, ymax - 0.08*(ymax-ymin)),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_v.text(x_label_pos_y, y_label_pos_y - 0.04*(ymax-ymin), v_axis,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=fontsize_axis_label)

    # ----- Unterer Plot (Stromvorhersage) -----
    ax_i.spines['left'].set_position('zero')
    ax_i.spines['bottom'].set_position('zero')
    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)
    ax_i.spines['left'].set_color(kit_dark_blue)
    ax_i.spines['bottom'].set_color(kit_dark_blue)
    ax_i.spines['left'].set_linewidth(line_size)
    ax_i.spines['bottom'].set_linewidth(line_size)

    # ----- Berechnung der Bereiche mit |v| < speed_threshold -----
    v_abs = np.abs(data[v_colname])
    low_speed_mask = v_abs < v_threshold
    diff = np.diff(low_speed_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if low_speed_mask.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if low_speed_mask.iloc[-1]:
        ends = np.append(ends, len(low_speed_mask) - 1)

    # ----- Einfärben der Bereiche mit |v| < speed_threshold, abhängig von z -----
    alpha = 0.2
    for start, end in zip(starts, ends):
        ax_i.axvspan(time[start], time[end], color=kit_green, alpha=alpha, linewidth=0)

    # Plot für Vorhersagen (dynamisch basierend auf y_configs)
    lines_pred = []
    if y_configs is None:
        y_configs = []

    for i, config in enumerate(y_configs):
        color = y_colors[i % len(y_colors)]
        line, _ = plot_prediction_with_std(
            data,
            config['ycolname'],
            color,
            config['ylabel']
        )
        if line is not None:
            lines_pred.append(line)

    # Plot für Strom-Messwerte
    line_i, = ax_i.plot(time, data[col_name], label='Strom-Messwerte',
                       color=kit_blue, linewidth=2)

    # Grid und Achsenbeschriftung
    ax_i.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=line_size/2)
    ax_i.set_axisbelow(True)
    ax_i.tick_params(axis='both', colors=kit_dark_blue, labelsize=fontsize_axis_label)

    # X-Achsenbeschriftung (nur beim unteren Plot)
    xmin, xmax = ax_i.get_xlim()
    ymin, ymax = ax_i.get_ylim()
    x_pos = -0.08 * (xmax - xmin)
    y_pos = -0.15 * ymax
    arrow_length = 0.04 * (xmax - xmin)

    ax_i.set_xlim(left=min(x_pos, xmin), right=xmax + arrow_length/2) # Achsenbegrenzungen anpassen

    ax_i.annotate('', xy=(xmax + arrow_length/2, 0), xytext=(xmax - arrow_length/2, 0),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_i.text(xmax*0.95, y_pos, r'$t$ in s',
             ha='left', va='center', color=kit_dark_blue, fontsize=fontsize_axis_label)

    # Y-Achsenbeschriftung
    y_pos = ymax * 0.85
    arrow_length = 0.04*(ymax-ymin)

    ax_i.set_ylim(bottom=min(y_pos, ymin), top=ymax + arrow_length / 2) # Achsenbegrenzungen anpassen

    ax_i.annotate('', xy=(0, ymax + arrow_length/2),
                 xytext=(0, ymax - arrow_length/2),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_i.text(x_pos, y_pos - 0.04*(ymax-ymin), label,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=fontsize_axis_label)

    # Titel über beiden Plots
    fig.suptitle(title, color=kit_dark_blue, fontsize=fontsize_title,
                fontweight='bold', y=0.98)

    # Legende (inkl. Näherungslinien)
    legend_elements = [line_i, line_v] + lines_pred
    legend_labels = [line.get_label() for line in legend_elements]

    # Farbige Bereiche zur Legende hinzufügen
    legend_elements.extend([
        Patch(facecolor=kit_green, alpha=alpha, label=f'Bereiche mit |v| < {v_threshold}'),
    ])
    legend_labels.extend([
        f'|v| < {v_threshold}',
    ])

    if len(legend_elements) > 1:
        fig.legend(
            handles=legend_elements,
            labels=legend_labels,
            loc='lower center',
            ncol=2,  # 4 Spalten für bessere Lesbarkeit
            frameon=True,
            facecolor='white',
            edgecolor=kit_dark_blue,
            framealpha=1.0,
            fontsize = fontsize_axis_label,
        )

    # Speichern des Plots
    plot_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    fig.savefig(plot_path + '.svg', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(plot_path + '.pdf', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved as {plot_path}')

def plot_time_series_error_with_sections(
    data: pd.DataFrame,
    title: str,
    filename: str,
    dpi: int = 300,
    col_name: str = 'curr_x',
    label: str = '$I$\nin A',
    y_configs: List[Dict[str, str]] = None,
    f_a: int = 50,
    path: str = 'Plots',
    v_colname: str = 'v_x',
    v_label: str = 'Vorschubgeschwindigkeit',
    v_axis: str = 'v in m/s',
    v_threshold: float = 1.0,
    scale_error: float = 2,
) -> None:
    fontsize_axis = 14
    fontsize_axis_label = 16
    fontsize_title = 18
    line_size = 1.5

    # KIT-Farben
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_yellow = "#FFFF00"
    kit_orange = "#FFC000"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_magenta = "#A3107C"
    kit_gray = "#767676"

    # Standardfarben für zusätzliche y-Achsen (kann erweitert werden)
    y_colors = [kit_red, kit_orange, kit_magenta, kit_yellow, kit_green]

    time = data.index / f_a

    # Erstelle Figure mit zwei Achsen
    fig, (ax_v, ax_i) = plt.subplots(
        2, 1, figsize=(12, 10), dpi=dpi,
        sharex=True, height_ratios=[1, 2],
        gridspec_kw={'hspace': 0.05}
    )

    # ----- Oberer Plot (Vorschubgeschwindigkeit) -----
    ax_v.spines['left'].set_position('zero')
    ax_v.spines['bottom'].set_position('zero')
    ax_v.spines['top'].set_visible(False)
    ax_v.spines['right'].set_visible(False)
    ax_v.spines['left'].set_color(kit_dark_blue)
    ax_v.spines['bottom'].set_color(kit_dark_blue)
    ax_v.spines['left'].set_linewidth(1.0)
    ax_v.spines['bottom'].set_linewidth(1.0)

    # Plot Vorschubgeschwindigkeit
    line_v, = ax_v.plot(time, data[v_colname], label=v_label,
                        color=kit_blue, linewidth=2)

    # Grid und Achsenbeschriftung
    ax_v.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5)
    ax_v.set_axisbelow(True)
    ax_v.tick_params(axis='both', colors=kit_dark_blue)

    # Achsenbeschriftung
    xmin, xmax = ax_v.get_xlim()
    ymin, ymax = ax_v.get_ylim()
    y_pos = -0.08 * ymax

    # X-Achsenbeschriftung
    ax_v.annotate('', xy=(xmax, 0), xytext=(xmax * 0.95, 0),
                  arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_v.text(xmax * 0.95, y_pos, r'$t$ in s',
              ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Y-Achsenbeschriftung
    x_label_pos_y = -0.06 * (xmax - xmin)
    y_label_pos_y = ymax * 0.65
    ax_v.annotate('', xy=(0, ymax),
                  xytext=(0, ymax - 0.08 * (ymax - ymin)),
                  arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_v.text(x_label_pos_y, y_label_pos_y - 0.04 * (ymax - ymin), v_axis,
              ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # ----- Unterer Plot (Stromvorhersage) -----
    ax_i.spines['left'].set_position('zero')
    ax_i.spines['bottom'].set_position('zero')
    ax_i.spines['top'].set_visible(False)
    ax_i.spines['right'].set_visible(False)
    ax_i.spines['left'].set_color(kit_dark_blue)
    ax_i.spines['bottom'].set_color(kit_dark_blue)
    ax_i.spines['left'].set_linewidth(line_size)
    ax_i.spines['bottom'].set_linewidth(line_size)

    # ----- Berechnung der Bereiche mit |v| < speed_threshold -----
    v_abs = np.abs(data[v_colname])
    low_speed_mask = v_abs < v_threshold
    diff = np.diff(low_speed_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if low_speed_mask.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if low_speed_mask.iloc[-1]:
        ends = np.append(ends, len(low_speed_mask) - 1)

    # ----- Einfärben der Bereiche mit |v| < speed_threshold, abhängig von z -----
    alpha = 0.2
    for start, end in zip(starts, ends):
        ax_i.axvspan(time[start], time[end], color=kit_green, alpha=alpha, linewidth=0)

    # Plot für Abweichungen (Mittelwert der Vorhersagen - tatsächlicher Strom)
    lines_pred = []
    if y_configs is None:
        y_configs = []
    for i, config in enumerate(y_configs):
        color = y_colors[i % len(y_colors)]
        # Finde alle Spalten, die zu dieser Konfiguration gehören
        cols = [col for col in data.columns if col.startswith(config['ycolname'])]
        if not cols:
            continue  # Überspringe, falls keine passenden Spalten gefunden wurden
        # Berechne Mittelwert der Vorhersagen
        mean_pred = data[cols].mean(axis=1)
        # Berechne Differenz: Mittelwert - tatsächlicher Strom
        deviation = mean_pred - data[col_name]
        # Temporäre Spalte für die Differenz hinzufügen
        data[f'deviation_{config["ycolname"]}'] = deviation * scale_error
        # Plot der Differenz
        line, _ = plot_prediction_with_std(
            data,
            f'deviation_{config["ycolname"]}',
            color,
            f'Abweichung {config["ylabel"]} (skaliert um {scale_error})'
        )
        if line is not None:
            lines_pred.append(line)

    # Plot für Strom-Messwerte
    line_i, = ax_i.plot(time, data[col_name], label='Strom-Messwerte',
                       color=kit_blue, linewidth=2)

    # Grid und Achsenbeschriftung
    ax_i.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=line_size/2)
    ax_i.set_axisbelow(True)
    ax_i.tick_params(axis='both', colors=kit_dark_blue, labelsize=fontsize_axis_label)

    # X-Achsenbeschriftung (nur beim unteren Plot)
    xmin, xmax = ax_i.get_xlim()
    ymin, ymax = ax_i.get_ylim()
    x_pos = -0.08 * (xmax - xmin)
    y_pos = -0.15 * ymax
    arrow_length = 0.04 * (xmax - xmin)

    ax_i.set_xlim(left=min(x_pos, xmin), right=xmax + arrow_length/2) # Achsenbegrenzungen anpassen

    ax_i.annotate('', xy=(xmax + arrow_length/2, 0), xytext=(xmax - arrow_length/2, 0),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_i.text(xmax*0.95, y_pos, r'$t$ in s',
             ha='left', va='center', color=kit_dark_blue, fontsize=fontsize_axis)

    # Y-Achsenbeschriftung
    y_pos = ymax * 0.85
    arrow_length = 0.04*(ymax-ymin)

    ax_i.set_ylim(bottom=min(y_pos, ymin), top=ymax + arrow_length / 2) # Achsenbegrenzungen anpassen

    ax_i.annotate('', xy=(0, ymax + arrow_length/2),
                 xytext=(0, ymax - arrow_length/2),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_i.text(x_pos, y_pos - 0.04*(ymax-ymin), label,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=fontsize_axis)

    # Titel über beiden Plots
    fig.suptitle(title, color=kit_dark_blue, fontsize=fontsize_title,
                fontweight='bold', y=0.98)

    # Legende (inkl. Näherungslinien)
    legend_elements = [line_i, line_v] + lines_pred
    legend_labels = [line.get_label() for line in legend_elements]

    # Farbige Bereiche zur Legende hinzufügen
    legend_elements.extend([
        Patch(facecolor=kit_green, alpha=alpha, label=f'Bereiche mit |v| < {v_threshold}'),
    ])
    legend_labels.extend([
        f'|v| < {v_threshold}',
    ])

    if len(legend_elements) > 1:
        fig.legend(
            handles=legend_elements,
            labels=legend_labels,
            loc='lower center',
            ncol=2,  # 4 Spalten für bessere Lesbarkeit
            frameon=True,
            facecolor='white',
            edgecolor=kit_dark_blue,
            framealpha=1.0,
            fontsize = fontsize_axis_label,
        )

    # Speichern des Plots
    plot_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    fig.savefig(plot_path + '.svg', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(plot_path + '.pdf', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved as {plot_path}')


def plot_prediction_with_std(data, base_label, color, label=''):
    """
    Hilfsfunktion zum Plotten von Vorhersagen mit Standardabweichung.
    """
    cols = [col for col in data.columns if col.startswith(base_label)]
    if not cols:
        return None, None

    mean = data[cols].mean(axis=1)
    std = data[cols].std(axis=1)
    line, = plt.gca().plot(data.index / 50, mean, label=label, color=color, linewidth=2)
    plt.gca().fill_between(data.index / 50, mean - std, mean + std, color=color, alpha=0.2)
    return line, mean
