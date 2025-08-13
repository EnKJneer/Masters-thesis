import os
from abc import abstractmethod, ABC
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from matplotlib.colors import LinearSegmentedColormap
from Visualization.Visualization_Heatmap import BasePlotter


class ModelComparisonHeatmapPlotter(BasePlotter):
    """Erstellt eine Vergleichs-Heatmap für mehrere Modelle mit MAE und Standardabweichung"""

    def __init__(self, output_dir: str, known_material: str = 'AL_2007_T4', known_geometry: str = 'Plate'):
        super().__init__(output_dir)
        self.known_material = known_material
        self.known_geometry = known_geometry

        # KIT-Farben definieren
        self.kit_red = "#D30015"
        self.kit_green = "#009682"
        self.kit_yellow = "#FFFF00"
        self.kit_orange = "#FFC000"
        self.kit_blue = "#0C537E"
        self.kit_magenta = "#A3107C"
        self.kit_dark_blue = "#002D4C"

        # Benutzerdefinierte Farbpalette erstellen (grün=gut, gelb=mittel, rot=schlecht)
        self.custom_cmap = LinearSegmentedColormap.from_list(
            'kit_colors',
            [self.kit_green, self.kit_yellow, self.kit_red],
            N=256
        )

    def get_text_color_for_background(self, mae_value):
        """Bestimmt die Textfarbe basierend auf dem Hintergrund (dunkel=weiß, hell=kit_dark_blue)"""
        if pd.isna(mae_value):
            return self.kit_dark_blue

        # Bei Werten < 0.5 (dunkler Hintergrund) verwende kit_dark_blue, sonst weiß
        if 0.1 < mae_value < 0.2:
            return self.kit_dark_blue
        else:
            return 'white'


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

    def get_model_seed_columns(self, df_columns: list, model_base_name: str):
        """Findet alle Seed-Spalten für ein bestimmtes Modell"""
        seed_columns = [col for col in df_columns if col.startswith(f'{model_base_name}_seed_')]
        if len(seed_columns) == 0:
            seed_columns = [col for col in df_columns if col == model_base_name]
        return seed_columns

    def calculate_mae_and_std_for_file(self, file_path: str, model_columns: list):
        """Berechnet MAE und Standardabweichung für alle Modelle in einer Datei"""
        try:
            df = pd.read_csv(file_path)
            print(f"Debug: Spalten in {os.path.basename(file_path)}: {df.columns.tolist()}")
            results = {}

            for model_base in model_columns:
                # Finde alle Seed-Spalten für dieses Modell
                seed_columns = self.get_model_seed_columns(df.columns, model_base)

                if len(seed_columns) > 0 and 'GroundTruth' in df.columns:
                    print(f"Debug: {model_base} - Gefundene Seed-Spalten: {seed_columns}")

                    # Berechne MAE für jede Seed-Spalte
                    mae_values = []
                    predictions_per_timestep = []

                    for seed_col in seed_columns:
                        # Entferne NaN-Werte
                        valid_indices = ~(df['GroundTruth'].isna() | df[seed_col].isna())
                        valid_count = valid_indices.sum()

                        if valid_count > 0:
                            mae = mean_absolute_error(
                                df.loc[valid_indices, 'GroundTruth'],
                                df.loc[valid_indices, seed_col]
                            )
                            mae_values.append(mae)

                            # Sammle Vorhersagen für Std-Berechnung
                            predictions_per_timestep.append(df.loc[valid_indices, seed_col].values)

                    if len(mae_values) > 0:
                        # Mittlerer MAE über alle Seeds
                        mean_mae = np.mean(mae_values)

                        # Standardabweichung der MAE-Werte zwischen den Seeds
                        std_mae = np.std(mae_values, ddof=1) if len(mae_values) > 1 else 0.0

                        # Alternative: Standardabweichung der Vorhersagen pro Zeitschritt
                        if len(predictions_per_timestep) > 1:
                            predictions_array = np.array(predictions_per_timestep)  # Shape: (n_seeds, n_timesteps)
                            std_predictions = np.mean(np.std(predictions_array, axis=0, ddof=1))
                        else:
                            std_predictions = 0.0

                        results[model_base] = {
                            'MAE': mean_mae,
                            'STD_MAE': std_mae,
                            'STD_PREDICTIONS': std_predictions,
                            'N_SEEDS': len(mae_values)
                        }

                        print(
                            f"Debug: {model_base} - MAE: {mean_mae:.4f}, STD_MAE: {std_mae:.4f}, STD_PRED: {std_predictions:.4f}, N_SEEDS: {len(mae_values)}")
                    else:
                        results[model_base] = {
                            'MAE': np.nan,
                            'STD_MAE': np.nan,
                            'STD_PREDICTIONS': np.nan,
                            'N_SEEDS': 0
                        }
                        print(f"Debug: {model_base} - Keine gültigen Werte")
                else:
                    results[model_base] = {
                        'MAE': np.nan,
                        'STD_MAE': np.nan,
                        'STD_PREDICTIONS': np.nan,
                        'N_SEEDS': 0
                    }
                    print(f"Debug: {model_base} - Keine Seed-Spalten gefunden oder GroundTruth fehlt")

        except Exception as e:
            print(f"Debug: Fehler beim Lesen von {file_path}: {e}")
            results = {model_base: {
                'MAE': np.nan,
                'STD_MAE': np.nan,
                'STD_PREDICTIONS': np.nan,
                'N_SEEDS': 0
            } for model_base in model_columns}

        return results

    def create_mae_dataframe(self, folder_path: str, model_columns: list):
        """Erstellt einen DataFrame mit MAE-Werten und Standardabweichungen für alle Dateien"""
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        results = []

        print(f"Debug: Gefundene Dateien: {files}")

        for file in files:
            file_path = os.path.join(folder_path, file)
            material, geometry = self.parse_filename(file)
            model_results = self.calculate_mae_and_std_for_file(file_path, model_columns)

            for model, metrics in model_results.items():
                results.append({
                    'Filename': file,
                    'Material': material,
                    'Geometry': geometry,
                    'Model': model,
                    'MAE': metrics['MAE'],
                    'STD_MAE': metrics['STD_MAE'],
                    'STD_PREDICTIONS': metrics['STD_PREDICTIONS'],
                    'N_SEEDS': metrics['N_SEEDS']
                })
                print(
                    f"Debug: {file} -> {material}/{geometry}/{model} -> MAE: {metrics['MAE']:.4f}, STD: {metrics['STD_PREDICTIONS']:.4f}")

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

    def format_cell_annotation(self, mae_value, std_value):
        """Formatiert die Zellenannotation mit MAE und Standardabweichung"""
        if pd.isna(mae_value) or pd.isna(std_value):
            return ""
        if std_value == 0 or std_value < 0.001:
            return f"{mae_value:.3f}"
        return f"{mae_value:.3f}\n±{std_value:.3f}"

    def create_dataset_names(self, mae_df):
        """Erstellt kombinierte Dataset-Namen aus Material und Geometrie"""
        mae_df['Dataset'] = mae_df['Material'] + '_' + mae_df['Geometry']
        return mae_df

    def create_plots(self, folder_path: str, model_columns: list, titel: str = 'Model Vergleich',
                     model_names=None, **kwargs):
        """Erstellt eine Vergleichs-Heatmap für alle Modelle"""
        # MAE-Daten erstellen
        mae_df = self.create_mae_dataframe(folder_path, model_columns)

        # Dataset-Namen erstellen
        mae_df = self.create_dataset_names(mae_df)

        # Unique Materialien und Geometrien ermitteln
        materials = mae_df['Material'].unique()
        geometries = mae_df['Geometry'].unique()

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
        for model in model_columns:
            clean_name = model.replace('Plate_TrainVal_', '').replace('ST_Data_', '').replace('ST_Plate_Notch_',
                                                                                              '').replace('_', ' ')
            model_clean_names[model] = clean_name

        # Pivot-Tabellen für MAE und STD erstellen
        mae_pivot = mae_df.pivot_table(
            values='MAE',
            index='Dataset',
            columns='Model',
            aggfunc='mean'
        )

        std_pivot = mae_df.pivot_table(
            values='STD_PREDICTIONS',
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

        titlesize = 30
        textsize = 25
        labelsize = 20

        # Heatmap mit seaborn für bessere Optik
        sns.heatmap(
            mae_pivot,
            annot=annotations,
            fmt='',  # Leerer Format-String, da wir custom annotations verwenden
            cmap=self.custom_cmap,
            mask=mask,
            cbar_kws={'label': 'MAE'},
            ax=ax,
            square=False,  # Nicht quadratisch, da verschiedene Anzahl von Zeilen/Spalten
            vmin=0.04,  # Minimum-Wert für Colorbar
            vmax=0.31,
            linewidths=0,  # Linien zwischen Zellen
            linecolor='white',
            annot_kws={'size': textsize, 'weight': 'bold', 'ha': 'center', 'va': 'center'}
        )

        # Textfarben für jede Zelle individuell setzen
        for i in range(mae_pivot.shape[0]):
            for j in range(mae_pivot.shape[1]):
                if not pd.isna(mae_pivot.iloc[i, j]):
                    text = ax.texts[i * mae_pivot.shape[1] + j]
                    text.set_color(text_colors[i, j])

        # Titel und Labels mit größerer Schrift
        ax.set_title(f'MAE Heatmap: {titel}', fontsize=titlesize, fontweight='bold', pad=20, color=self.kit_dark_blue)
        ax.set_xlabel('Model', fontsize=textsize, fontweight='bold', color=self.kit_dark_blue)
        ax.set_ylabel('Datensatz', fontsize=textsize, fontweight='bold', color=self.kit_dark_blue)

        # Achsenbeschriftungen vergrößern und Farbe setzen
        ax.tick_params(axis='both', which='major', labelsize=labelsize, colors=self.kit_dark_blue)

        # Achsenbeschriftungen (Tick-Labels) explizit färben
        for label in ax.get_xticklabels():
            label.set_color(self.kit_dark_blue)
        for label in ax.get_yticklabels():
            label.set_color(self.kit_dark_blue)

        # Colorbar-Label vergrößern
        cbar = ax.collections[0].colorbar
        cbar.set_label('MAE', fontsize=textsize, fontweight='bold', color=self.kit_dark_blue)
        cbar.ax.tick_params(labelsize=labelsize, colors=self.kit_dark_blue)

        # Colorbar Tick-Labels explizit färben
        for label in cbar.ax.get_yticklabels():
            label.set_color(self.kit_dark_blue)

        plt.tight_layout()

        # Speichern
        filename = 'RNN_heatmap_model_comparison_with_std.png'
        plot_path = self.save_plot(fig, filename)

        print(f"Model Comparison Heatmap mit Standardabweichung erstellt: {plot_path}")
        return [plot_path]


if __name__ == '__main__':
    # Konfiguration
    folder_result = 'Plots'
    # folder = '..\\Experiment\\Controller_Input/Results/ST_Plate_Notch-2025_07_29_18_25_46/Predictions'
    # folder = '..\\Experiment\\Hyperparameteroptimization/Results/Random_Forest/2025_07_28_14_40_41/Predictions'
    folder = '..\\Experiment\\Hyperparameteroptimization/Results/Recurrent_Neural_Net/2025_07_28_19_20_29/Predictions'

    known_material = 'S235JR'
    known_geometry = 'Plate'

    # Modelle definieren (Base-Namen ohne _seed_X)

    models = ['ST_Plate_Notch_Recurrent_Neural_Net', 'ST_Plate_Notch_Recurrent_Neural_Net_TPESampler',
              'ST_Plate_Notch_Recurrent_Neural_Net_RandomSampler', 'ST_Plate_Notch_Recurrent_Neural_Net_GridSampler']
    '''
    models = ['ST_Plate_Notch_Random_Forest', 'ST_Plate_Notch_Random_Forest_TPESampler',
              'ST_Plate_Notch_Random_Forest_RandomSampler', 'ST_Plate_Notch_Random_Forest_GridSampler']
    '''
    # models = ['ohne cont_dev_Recurrent_Neural_Net', 'mit cont_dev_Recurrent_Neural_Net']
    # Plotter erstellen
    plotter = ModelComparisonHeatmapPlotter(
        output_dir=folder_result,
        known_material=known_material,
        known_geometry=known_geometry
    )

    names = {
        'ST_Plate_Notch_Random_Forest': 'Initial',
        'ST_Plate_Notch_Random_Forest_TPESampler': 'TPE Sampler',
        'ST_Plate_Notch_Random_Forest_RandomSampler': 'Random Sampler',
        'ST_Plate_Notch_Random_Forest_GridSampler': 'Grid Sampler',

        'ST_Plate_Notch_Recurrent_Neural_Net': 'Initial',
        'ST_Plate_Notch_Recurrent_Neural_Net_TPESampler': 'TPE Sampler',
        'ST_Plate_Notch_Recurrent_Neural_Net_RandomSampler': 'Random Sampler',
        'ST_Plate_Notch_Recurrent_Neural_Net_GridSampler': 'Grid Sampler',

        'mit cont_dev_Recurrent_Neural_Net': 'mit Regler',
        'ohne cont_dev_Recurrent_Neural_Net': 'ohne Regler',
        'mit cont_dev_Random_Forest': 'mit Regler',
        'ohne cont_dev_Random_Forest': 'ohne Regler'
    }

    # Plots erstellen
    try:
        plot_paths = plotter.create_plots(folder, models, titel='Random Forest', model_names=names)
        print(f"\nAlle Heatmaps wurden erfolgreich erstellt!")
        print(f"Gespeichert in: {folder_result}")
        for path in plot_paths:
            print(f"  - {path}")
    except Exception as e:
        print(f"Fehler beim Erstellen der Plots: {e}")
        import traceback