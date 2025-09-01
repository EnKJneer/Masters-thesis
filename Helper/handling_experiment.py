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
from numpy.exceptions import AxisError
from sklearn.metrics import mean_absolute_error

import Helper.handling_data as hdata
import Models.model_neural_net as mnn
import Models.model_random_forest as mrf
from matplotlib.colors import LinearSegmentedColormap

HEADER = ["DataSet", "DataPath", "Model", "MAE", "StdDev", "MAE_Ensemble", "Predictions", "GroundTruth", "RawData"]
SAMPLINGRATE = 50
AXIS = 'x'

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
        self.colors = [self.kit_red, self.kit_yellow, self.kit_orange, self.kit_magenta, self.kit_green]

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
        name_without_ext = name_without_ext.replace('DMC_', '')
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

class ModelComparisonPlotter(BasePlotter):
    """Erstellt eine Übersichtsplot aller Modelle über alle Dataset/DataPath Kombinationen"""


    def create_plots(self, df: pd.DataFrame, title: str = 'Model Vergleich', model_names=None, **kwargs):
        """
        Erstellt einen Balkenplot-Vergleich für alle Modelle mit MAE und Standardabweichung.
        Args:
            df: DataFrame mit den Daten
            title: Titel des Plots
            model_names: Dictionary für benutzerdefinierte Modellnamen
            **kwargs: Weitere Parameter
        """
        # Daten vorbereiten
        df = df.copy()
        df[['Material', 'Geometry']] = df['DataPath'].apply(lambda x: pd.Series(self.parse_filename(x)))
        df['Dataset'] = df['Material'] + '_' + df['Geometry']

        # Kategorien ordnen
        dataset_paths = df['Dataset'].unique()

        materials, geometries = set(), set()

        # Material und Geometrie aus Dataset_Path extrahieren
        for path in dataset_paths:
            material, geometry = self.parse_filename(path)
            materials.add(material)
            geometries.add(geometry)

        ordered_datasets = self.create_ordered_categories_from_datapath(dataset_paths)

        # Modelle bereinigen
        models = df['Model'].unique()
        model_clean_names = {}
        for model in models:
            clean_name = model.replace('Plate_TrainVal_', '').replace('Reference_', '').replace('ST_Data_', '') \
                             .replace('ST_Plate_Notch_', '').replace('Ref', '').replace('_', ' ')
            model_clean_names[model] = clean_name

        # Pivot-Tabellen für MAE und STD erstellen
        mae_pivot = df.pivot_table(values='MAE', index='Dataset', columns='Model', aggfunc='mean')
        std_pivot = df.pivot_table(values='StdDev', index='Dataset', columns='Model', aggfunc='mean')

        # Reorder basierend auf der gewünschten Reihenfolge
        mae_pivot = mae_pivot.reindex(index=ordered_datasets)
        std_pivot = std_pivot.reindex(index=ordered_datasets)

        # Spalten umbenennen
        if model_names is None:
            mae_pivot.columns = [model_clean_names.get(col, col) for col in mae_pivot.columns]
            std_pivot.columns = [model_clean_names.get(col, col) for col in std_pivot.columns]
        else:
            mae_pivot.columns = [model_names.get(col, col) for col in mae_pivot.columns]
            std_pivot.columns = [model_names.get(col, col) for col in std_pivot.columns]

        # Plot erstellen
        fig, ax = plt.subplots(figsize=(14, 12))
        fig.set_dpi(1200)

        # Balkenbreite und Positionen
        x = np.arange(len(ordered_datasets))
        bar_width = 0.8 / len(models) if len(models) > 0 else 0.8

        # Farben zuweisen
        colors = [self.get_model_color(model) or self.colors[i % len(self.colors)] for i, model in enumerate(models)]

        # Balken plotten
        for i, model in enumerate(mae_pivot.columns):
            y = mae_pivot[model].values
            yerr = std_pivot[model].values
            x_pos = x + i * bar_width
            bars = ax.bar(x_pos, y, width=bar_width, label=model, yerr=yerr, capsize=4,
                          color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)

            # Annotationen hinzufügen
            for bar, mae_val, std_val in zip(bars, y, yerr):
                if not pd.isna(mae_val):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                            f"{mae_val:.3f}\n±{std_val:.3f}" if std_val > 0.001 else f"{mae_val:.3f}",
                            ha='center', va='bottom', fontsize=10, fontweight='bold', color=self.kit_dark_blue)

        # Titel und Labels
        titlesize = 30
        labelsize = 25
        textsize = 18

        ax.set_title(f'MAE Vergleich: {title}', fontsize=titlesize, fontweight='bold', pad=20, color=self.kit_dark_blue)
        ax.set_xlabel('Datensatz', fontsize=labelsize, fontweight='bold', color=self.kit_dark_blue)
        ax.set_ylabel('MAE $I$ in A', fontsize=labelsize, fontweight='bold', color=self.kit_dark_blue)

        # x-Achsen-Tick-Labels anpassen
        y_labels = []
        for dataset in ordered_datasets:
            material, geometry = self.parse_filename(dataset)
            material = self.replace_material_name(material)
            y_labels.append(f"{material}\n{geometry}")

        ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
        ax.set_xticklabels(y_labels, fontsize=textsize, color=self.kit_dark_blue, rotation=45, ha='right')

        # Legende und Grid
        ax.legend(title='Modelle', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=textsize)
        ax.grid(True, alpha=0.3, linestyle='--', color='gray')

        # Achsenbeschriftungen färben
        ax.tick_params(axis='both', which='major', labelsize=textsize, colors=self.kit_dark_blue)

        plt.tight_layout()

        # Speichern
        filename = 'model_comparison_bars.png'
        plot_path = self.save_plot(fig, filename)
        print(f"Model Comparison Bars erstellt: {plot_path}")
        return [plot_path]

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
        # Unique Materialien, Geometrien und Modelle ermitteln
        materials = mae_df['Material'].unique()
        geometries = mae_df['Geometry'].unique()
        models = mae_df['Model'].unique()
        # Kategorien ordnen
        materials_ordered, geometries_ordered = self.create_ordered_categories(materials, geometries)
        # Datasets in gewünschter Reihenfolge erstellen
        ordered_datasets = []
        for material in materials_ordered:
            for geometry in geometries_ordered:
                dataset_name = f"{material}_{geometry}"
                if dataset_name in mae_df['Dataset'].unique():
                    ordered_datasets.append(dataset_name)
        # Model-Namen für bessere Lesbarkeit bereinigen
        model_clean_names = {}
        for model in models:
            clean_name = model.replace('Plate_TrainVal_', '').replace('Reference_', '').replace('ST_Data_', '') \
                             .replace('ST_Plate_Notch_', '').replace('Ref', '').replace('_', ' ')
            model_clean_names[model] = clean_name
        # Pivot-Tabellen für MAE und STD erstellen
        mae_pivot = mae_df.pivot_table(
            values='MAE',
            index='Dataset',
            columns='Model',
            aggfunc='mean'
        )
        std_pivot = mae_df.pivot_table(
            values='StdDev',
            index='Dataset',
            columns='Model',
            aggfunc='mean'
        )
        # Reorder basierend auf der gewünschten Reihenfolge
        mae_pivot = mae_pivot.reindex(index=ordered_datasets)
        std_pivot = std_pivot.reindex(index=ordered_datasets)
        # Spalten umbenennen für bessere Lesbarkeit
        if model_names is None:
            mae_pivot.columns = [model_clean_names.get(col, col) for col in mae_pivot.columns]
            std_pivot.columns = [model_clean_names.get(col, col) for col in std_pivot.columns]
        else:
            mae_pivot.columns = [model_names.get(col, col) for col in mae_pivot.columns]
            std_pivot.columns = [model_names.get(col, col) for col in std_pivot.columns]
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
        maesize = 35
        textsize = 25
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
        # Titel und Labels mit größerer Schrift
        ax.set_title(f'MAE Heatmap: {title}', fontsize=titlesize, fontweight='bold', pad=20, color=self.kit_dark_blue)
        ax.set_xlabel('Model', fontsize=labelsize, fontweight='bold', color=self.kit_dark_blue)
        ax.set_ylabel('Datensatz', fontsize=labelsize, fontweight='bold', color=self.kit_dark_blue)

        # Achsenbeschriftungen vergrößern und Farbe setzen
        ax.tick_params(axis='both', which='major', labelsize=textsize, colors=self.kit_dark_blue)

        # Hier: y-Achsen-Tick-Labels anpassen
        y_labels = []
        for dataset in ordered_datasets:
            material, geometry = self.parse_filename(dataset)
            # Materialnamen ersetzen (z. B. S235JR -> Stahl)
            material = self.replace_material_name(material)
            y_labels.append(f"{material}\n{geometry}")

        ax.set_yticklabels(y_labels, fontsize=textsize, color=self.kit_dark_blue, ha='center', va='center')


        # x-Achsen-Tick-Labels anpassen (Zeilenumbrüche hinzufügen)
        x_labels = []
        for model in mae_pivot.columns:
            parts = model.split()
            if len(parts) > 2:
                new_parts = []
                for i in range(0, len(parts), 2):
                    new_parts.append(' '.join(parts[i:i + 2]))
                model = '\n'.join(new_parts)
            x_labels.append(model)

        # Schriftgröße der x-Achsen-Tick-Labels reduzieren
        #x_labelsize = min(labelsize, 12)  # Dynamische Anpassung der Schriftgröße
        ax.set_xticklabels(x_labels, fontsize=textsize, color=self.kit_dark_blue, ha='center')

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
        filename = f'heatmap_model_comparison_with_std.png'
        plot_path = self.save_plot(fig, filename)
        print(f"Model Comparison Heatmap mit Standardabweichung erstellt: {plot_path}")
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
            ax.set_title(f'MAE Heatmap: {model_clean}', fontsize=titlesize, fontweight='bold', pad=20,
                         color=self.kit_dark_blue)
            ax.set_xlabel('Geometry', fontsize=labelsize, fontweight='bold', color=self.kit_dark_blue)
            ax.set_ylabel('Material', fontsize=labelsize, fontweight='bold', color=self.kit_dark_blue)

            materials_ordered = self.replace_material_names(materials_ordered)
            # Achsenbeschriftungen anpassen: (Bekannt)/(Unbekannt) hinzufügen
            x_labels = [f"{geo}\n(Bekannt)" if geo == self.known_geometry else f"{geo}\n(Unbekannt)" for geo in
                        geometries_ordered]
            y_labels = [f"{mat}\n(Bekannt)" if mat == self.known_material else f"{mat}\n(Unbekannt)" for mat in
                        materials_ordered]

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
            fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

            # DIN 461: Achsen müssen durch den Nullpunkt gehen
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.kit_dark_blue)
            ax.spines['bottom'].set_color(self.kit_dark_blue)
            ax.spines['left'].set_linewidth(1.0)
            ax.spines['bottom'].set_linewidth(1.0)

            time = np.arange(len(ground_truth)) / SAMPLINGRATE
            color_idx = 0

            for dataset in datasets:
                for model in models:
                    row = df_subset[
                        (df_subset['Model'] == model) &
                        (df_subset['DataSet'] == dataset)
                        ].iloc[0]

                    # Farbzuweisung basierend auf Schlüsselwörtern
                    #color = self.get_model_color(model)
                    #if color is None:
                    color = self.colors[color_idx % len(self.colors)]

                    # angenommen: hier steckt ein Array der Form (n_runs, time)
                    preds_list = np.array(row['Predictions'])
                    # Mittelwert und Standardabweichung über die Runs
                    mean_pred = preds_list.mean(axis=0)
                    std_pred = preds_list.std(axis=0)

                    # Konfidenzband
                    ax.fill_between(
                        time,
                        mean_pred - std_pred,
                        mean_pred + std_pred,
                        color=color,
                        alpha=0.2
                    )

                    if len(datasets) == 1:
                        label = f"{model}"
                    else:
                        label = f"{model} ({dataset})"

                    # Mittelwertkurve
                    ax.plot(
                        time,
                        mean_pred,
                        label=label,
                        color=color,
                        linewidth=2
                    )
                    color_idx += 1

            # Plot GroundTruth
            time = np.arange(len(ground_truth)) / SAMPLINGRATE
            ax.plot(time, ground_truth, label='Messwerte', color=self.kit_dark_blue, linewidth=2)

            # DIN 461: Beschriftungen in kit_dark_blue
            ax.tick_params(axis='x', colors=self.kit_dark_blue, direction='inout', length=6)
            ax.tick_params(axis='y', colors=self.kit_dark_blue, direction='inout', length=6)

            # Grid nach DIN 461
            ax.grid(True, color=self.kit_dark_blue, alpha=0.3, linewidth=0.5, linestyle='-')
            ax.set_axisbelow(True)

            # Achsenbeschriftungen mit Pfeilen
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            arrow_length = 0.03 * (xmax - xmin)
            arrow_height = 0.04 * (ymax - ymin)

            # X-Achse: Pfeil bei der Beschriftung
            x_label_pos = xmax
            y_label_pos = -0.08 * (ymax - ymin)
            ax.annotate('', xy=(x_label_pos + arrow_length, y_label_pos),
                        xytext=(x_label_pos, y_label_pos),
                        arrowprops=dict(arrowstyle='->', color=self.kit_dark_blue,
                                        lw=1.5, shrinkA=0, shrinkB=0,
                                        mutation_scale=15))
            ax.text(x_label_pos - 0.06 * (xmax - xmin), y_label_pos, r'$t$ in s',
                    ha='left', va='center', color=self.kit_dark_blue, fontsize=12)

            # Y-Achse: Pfeil bei der Beschriftung
            x_label_pos_y = -0.06 * (xmax - 0)
            y_label_pos_y = ymax * 0.85
            ax.annotate('', xy=(x_label_pos_y, y_label_pos_y + arrow_height),
                        xytext=(x_label_pos_y, y_label_pos_y),
                        arrowprops=dict(arrowstyle='->', color=self.kit_dark_blue,
                                        lw=1.5, shrinkA=0, shrinkB=0,
                                        mutation_scale=15))
            ax.text(x_label_pos_y, y_label_pos_y - 0.04 * (ymax - ymin), '$I$ in A',
                    ha='center', va='bottom', color=self.kit_dark_blue, fontsize=12)

            material, geometry = self.parse_filename(datapath)
            material = self.replace_material_name(material)

            # Titel mit DIN 461 konformer Positionierung
            ax.set_title(f"{material} {geometry}: Stromverlauf der Vorschubachse in {AXIS}-Richtung",
                         color=self.kit_dark_blue, fontsize=14, fontweight='bold', pad=20)

            # Legende
            legend = ax.legend(loc='upper right',
                               frameon=True, fancybox=False, shadow=False,
                               framealpha=1.0, facecolor='white', edgecolor=self.kit_dark_blue)
            legend.get_frame().set_linewidth(1.0)
            for text in legend.get_texts():
                text.set_color(self.kit_dark_blue)

            # DIN 461: Achsenbegrenzungen anpassen
            ax.set_xlim(left=min(x_label_pos_y, xmin), right=xmax * 1.05)
            ax.set_ylim(bottom=min(y_label_pos, ymin), top=ymax * 1.05)

            filename = f"predictions_vs_groundtruth_{datapath.replace('/', '_')}.png"
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
                #try:
                plot_paths = self.plotters[plotter_name].create_plots(df)
                all_plot_paths[plotter_name] = plot_paths
                print(f"✓ {plotter_name}: {len(plot_paths)} plots created")
                #except Exception as e:
                #    print(f"✗ Error creating {plotter_name}: {str(e)}")

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
    plot_manager.register_plotter('geometry_mae', MAEPlotterGeometry(plots_dir))
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

# In calculate_and_store_results Funktion:
def calculate_and_store_results(model, dataClass, nn_preds, y_test, df_list_results, results, header_list,
                                raw_data):  # dataSets Parameter hinzufügen
    """
    Calculate MAE and standard deviation, and store the results.
    """
    for j, path in enumerate(dataClass.testing_data_paths):
        name = model.name + "_" + path.replace('.csv', '')

        mse_nn, std_nn, mae_ensemble = calculate_mae_and_std(nn_preds[j],
                                                                   y_test[j].values if isinstance(y_test[j],
                                                                                                  pd.DataFrame) else
                                                                   y_test[j])

        predictions = []
        for pred in nn_preds[j]:
            predictions.append(pred.tolist())

        # Raw data als Dictionary speichern (JSON-serialisierbar)
        raw_data_dict = {
            'columns': raw_data[j].columns.tolist(),
            'data': raw_data[j].to_dict('records')  # Als Liste von Dictionaries
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
            y_test[j].values.tolist() if isinstance(y_test[j], pd.DataFrame) else y_test[j].tolist(),
            raw_data_dict  # RawData als Dictionary
        ])
        header_list[j].append(name)


def run_experiment(dataSets, models,
                   NUMBEROFEPOCHS=800, NUMBEROFMODELS=10, batched_data=False,
                   patience=5, plot_types=None,
                   experiment_name = 'Experiment'):

    if type(dataSets) is not list:
        dataSets = [dataSets]
    if type(models) is not list:
        models = [models]

    # Create directory for results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = experiment_name +'-'+ timestamp
    results_dir = os.path.join('Results', results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # Define the meta information structure
    meta_information = {
        "DataSets": [],
        "Models": [],
        "Data_Preprocessing": {
            "batched_data": batched_data,
            "NUMBEROFEPOCHS": NUMBEROFEPOCHS,
        }
    }

    # Add datasets to meta information
    for data_params in dataSets:
        data_info = {
            **data_params.get_documentation()
        }
        meta_information["DataSets"].append(data_info)

    """ Prediction """
    results = []
    models_copy = copy.deepcopy(models)
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
        for idx, model in enumerate(models):
            # Modellvergleich auf neuen Daten
            nn_preds = [[] for _ in range(len(X_test))] if isinstance(X_test, list) else []

            for _ in range(NUMBEROFMODELS):
                model = models_copy[idx]
                if hasattr(model, 'input_size'):
                    model.input_size = None
                    model.scaler = None
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
                models[idx] = model # save for documentation
            # Fehlerberechnung
            calculate_and_store_results(model, dataClass, nn_preds, y_test, df_list_results, results, header_list,
                                        raw_data)

    # After training to include learned parameters
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

def calculate_mae_and_std(predictions_list, true_values, n_drop_values=10, center_data = False):
    mae_values = []

    for pred in predictions_list:
        # Werte kürzen
        pred_trimmed = pred[n_drop_values:-n_drop_values]
        true_trimmed = true_values[n_drop_values:-n_drop_values]

        if center_data:
            # Zentrierung
            mean = np.mean(true_trimmed)
            pred_centered = pred_trimmed - mean
            true_centered = true_trimmed - mean
        else:
            pred_centered = pred_trimmed
            true_centered = true_trimmed

        mae = np.mean(np.abs(pred_centered.squeeze() - true_centered.squeeze()))
        mae_values.append(mae)
    pred_mean = np.mean(predictions_list, axis=0)
    mae_ensemble = np.mean(np.abs(pred_mean.squeeze() - true_values.squeeze()))

    return np.mean(mae_values), np.std(mae_values), mae_ensemble

def save_detailed_csv(df, results_dir):
    """
    Speichert detaillierte Daten für jeden DataPath in separaten CSV-Dateien.
    Jede Zeitreihe (Seed/Run) wird einzeln gespeichert anstatt als Mittelwert.

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

        # Erstelle einen DataFrame für alle individuellen Vorhersagen
        predictions_dict = {}

        for _, row in df_subset.iterrows():
            predictions = row['Predictions']  # Shape: (n_seeds, n_timesteps)
            dataset_model = f'{row["DataSet"]}_{row["Model"]}'

            # Füge jede einzelne Zeitreihe hinzu
            for seed_idx in range(len(predictions)):
                column_name = f'{dataset_model}_seed_{seed_idx}'
                predictions_dict[column_name] = predictions[seed_idx]

        # Erstelle einen DataFrame aus dem Wörterbuch der individuellen Vorhersagen
        predictions_df = pd.DataFrame(predictions_dict)

        # Kombiniere die DataFrames
        combined_df = pd.concat([raw_data_df, ground_truth_df, predictions_df], axis=1)

        # Speichere den kombinierten DataFrame in einer CSV-Datei
        csv_file = os.path.join(detailed_csv_dir, f'{datapath.replace("/", "_")}.csv')
        combined_df.to_csv(csv_file, index=False)

    print(f"Detaillierte CSV-Dateien mit individuellen Zeitreihen wurden in {detailed_csv_dir} gespeichert.")

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