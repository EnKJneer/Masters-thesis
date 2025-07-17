import os
from abc import abstractmethod, ABC
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from matplotlib.colors import LinearSegmentedColormap


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


if __name__ == '__main__':
    # Konfiguration
    folder_result = 'Plots'
    folder = 'Results/2025_07_13_12_04_29/Predictions'

    known_material = 'AL_2007_T4'
    known_geometry = 'Plate'

    # Modelle definieren
    models = ['Plate_TrainVal_Random_Forest', 'Plate_TrainVal_Neural_Net']

    # Plotter erstellen
    plotter = ModelHeatmapPlotter(
        output_dir=folder_result,
        known_material=known_material,
        known_geometry=known_geometry
    )

    # Plots erstellen
    try:
        plot_paths = plotter.create_plots(folder, models)
        print(f"\nAlle Heatmaps wurden erfolgreich erstellt!")
        print(f"Gespeichert in: {folder_result}")
        for path in plot_paths:
            print(f"  - {path}")
    except Exception as e:
        print(f"Fehler beim Erstellen der Plots: {e}")
        import traceback

        traceback.print_exc()