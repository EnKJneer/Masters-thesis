import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import Helper.handling_data as hdata
from matplotlib.colors import LinearSegmentedColormap

class CorrelationPlotter:
    """Erstellt Korrelationsplots mit KIT-Designrichtlinien und standardisierten Achsenbezeichnungen"""
    def __init__(self, output_dir="Plot_Correlation", velocity_threshold=None):
        self.output_dir = output_dir
        self.velocity_threshold = velocity_threshold  # Schwellwert für |v|
        os.makedirs(output_dir, exist_ok=True)
        # KIT Farbpalette
        self.kit_red = "#D30015"
        self.kit_green = "#009682"
        self.kit_yellow = "#FFFF00"
        self.kit_orange = "#FFC000"
        self.kit_blue = "#0C537E"
        self.kit_dark_blue = "#002D4C"
        self.kit_magenta = "#A3107C"
        # Standardisierte Achsenbezeichnungen und Titel
        self.column_metadata = {
            # Strom-Spalten
            'curr_x': {'axis': '$I$\nin A', 'title': 'Strom  in X-Richtung', 'unit': 'A'},
            'curr_y': {'axis': '$I$\nin A', 'title': 'Strom  in Y-Richtung', 'unit': 'A'},
            'curr_z': {'axis': '$I$\nin A', 'title': 'Strom  in Z-Richtung', 'unit': 'A'},
            'curr_sp': {'axis': '$I$\nin A', 'title': 'Spindelstrom', 'unit': 'A'},
            # Geschwindigkeits-Spalten
            'v_x': {'axis': '$v$\nin mm/s', 'title': 'Vorschubgeschwindigkeit', 'unit': 'mm/s'},
            'v_y': {'axis': '$v$\nin mm/s', 'title': 'Vorschubgeschwindigkeit', 'unit': 'mm/s'},
            'v_z': {'axis': '$v$\nin mm/s', 'title': 'Vorschubgeschwindigkeit', 'unit': 'mm/s'},
            'v_sp': {'axis': '$v$\nin mm/s', 'title': 'Spindelgeschwindigkeit', 'unit': 'mm/s'},
            # Kraft-Spalten (simuliert)
            'f_x_sim': {'axis': 'F\nin N', 'title': 'Prozesskraft', 'unit': 'N'},
            'f_y_sim': {'axis': 'F\nin N', 'title': 'Prozesskraft', 'unit': 'N'},
            'f_z_sim': {'axis': 'F\nin N', 'title': 'Prozesskraft', 'unit': 'N'},
            'f_sp_sim': {'axis': 'F\nin N', 'title': 'Spindelkraft', 'unit': 'N'},
            # Materialentfernung
            'materialremoved_sim': {
                'axis': 'MRR in mm³/s',
                'title': 'MRR',
                'unit': 'mm³/s'
            },
            # Berechnete Terme
            'Term_x': {
                'axis': 'Term\nin N·mm²',
                'title': 'Term X-Achse',
                'unit': 'N·mm²'
            },
            'Term_y': {
                'axis': 'Term\nin N·mm²',
                'title': 'Term Y-Achse',
                'unit': 'N·mm²'
            },
            'Term_z': {
                'axis': 'Term\nin N·mm²',
                'title': 'Term Z-Achse',
                'unit': 'N·mm²'
            },
            'Term_sp': {
                'axis': 'Term\nin N·mm²',
                'title': 'Term Spindel',
                'unit': 'N·mm²'
            }
        }
        # Standard-Plot-Stil konfigurieren
        self._configure_plot_style()

    def _configure_plot_style(self):
        """Konfiguriert den Standard-Plot-Stil"""
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.titlesize': 20,
            'figure.dpi': 120,
            'lines.linewidth': 2.5,
            'axes.grid': True,
            'grid.color': '0.8',
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white'
        })

    def _remove_outliers(self, df, column1, column2, threshold_std=12):
        """
        Entfernt Outlier aus dem DataFrame für die gegebenen Spalten.
        Outlier sind definiert als Werte, die mehr als threshold_std Standardabweichungen
        vom Mittelwert entfernt sind.
        """
        df_clean = df.copy()
        for col in [column1, column2]:
            if col in df_clean.columns:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                # Berechne die Grenzen für Outlier
                lower_bound = mean - threshold_std * std
                upper_bound = mean + threshold_std * std
                # Filtere Outlier
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        return df_clean

    def _filter_by_velocity_threshold(self, df, column1, column2):
        """
        Filtert Datenpunkte basierend auf dem Schwellwert für |v|.
        Falls column1 oder column2 eine Geschwindigkeitskomponente (v_*) ist,
        wird nur geplottet, wenn |v| < threshold.
        """
        if self.velocity_threshold is None:
            return df  # Kein Filter, wenn kein Schwellwert gesetzt

        # Prüfe, ob eine der Spalten eine Geschwindigkeitskomponente ist
        v_columns = [col for col in [column1, column2] if col.startswith('v_')]
        if not v_columns:
            return df  # Keine Geschwindigkeitsfilterung nötig

        # Filtere Datenpunkte, bei denen |v| >= threshold
        for v_col in v_columns:
            df = df[abs(df[v_col]) < self.velocity_threshold]
        return df

    def get_column_metadata(self, column_name):
        """Gibt die Metadaten für eine Spalte zurück oder Standardwerte"""
        if column_name in self.column_metadata:
            return self.column_metadata[column_name]
        return {
            'axis': column_name.replace('_', ' ').title(),
            'title': column_name.replace('_', ' ').title(),
            'unit': ''
        }

    def load_csv_files(self, csv_paths):
        """Lädt mehrere CSV-Dateien in ein DataFrame"""
        dfs = []
        for path in csv_paths:
            file_path = os.path.join(hdata.DataClass_ST_Plate_Notch.folder, path)
            df = pd.read_csv(file_path)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def _filter_zero_values(self, df, column1, column2):
        """
        Filtert Datenpunkte, bei denen einer der Werte 0 ist
        und gibt den gefilterten DataFrame zurück
        """
        # Erzeuge eine Maske für Nicht-Null-Werte in beiden Spalten
        mask = (df[column1] != 0) & (df[column2] != 0)
        # Filtere auch extrem kleine Werte nahe 0 (z.B. < 1e-6)
        mask &= (abs(df[column1]) > 1e-6) & (abs(df[column2]) > 1e-6)
        return df[mask]

    def _should_add_density(self, x_data, y_data):
        """Entscheidet, ob ein Dichteplot hinzugefügt werden soll"""
        # Nur hinzufügen wenn genug Datenpunkte vorhanden sind
        # und die Daten eine gewisse Variabilität aufweisen
        if len(x_data) < 100:
            return False
        x_var = np.var(x_data)
        y_var = np.var(y_data)
        return x_var > 1e-6 and y_var > 1e-6 and len(x_data) > 100

    def plot_correlation(self, df, column1, column2, output_filename=None):
        """
        Erstellt einen Korrelationsplot mit KIT-Design und speichert ihn
        Args:
            df: DataFrame mit den Daten
            column1: Name der ersten Spalte
            column2: Name der zweiten Spalte
            output_filename: Optionaler Dateiname für die Ausgabe
        """
        # Outlier entfernen
        df_clean = self._remove_outliers(df, column1, column2)
        # Nullwerte filtern
        df_filtered = self._filter_zero_values(df_clean, column1, column2)
        # Geschwindigkeitsfilter anwenden (falls Schwellwert gesetzt)
        df_filtered = self._filter_by_velocity_threshold(df_filtered, column1, column2)

        # Wenn nach dem Filtern zu wenige Datenpunkte übrig sind, abbrechen
        if len(df_filtered) < 10:
            print(f"Zu wenige Datenpunkte nach Filterung für {column1} vs {column2} ({len(df_filtered)} Punkte)")
            return None

        # Metadaten für die Spalten abrufen
        meta1 = self.get_column_metadata(column1)
        meta2 = self.get_column_metadata(column2)
        # Korrelation berechnen
        correlation = df_filtered[[column1, column2]].corr().iloc[0, 1]

        # Plot erstellen
        plt.figure(figsize=(14, 10))
        sns.set_style("whitegrid")

        # Scatterplot mit KIT-Farben
        scatter = sns.scatterplot(
            data=df_filtered,
            x=column1,
            y=column2,
            color=self.kit_blue,
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5,
            s=80
        )

        # **Keine Regressionsgerade mehr!** (Entfernt: sns.regplot)

        # Titel und Labels mit KIT-Farben und Metadaten
        correlation_text = f"Korrelation: {correlation:.3f}"
        if self.velocity_threshold is not None:
            correlation_text += f" (|v| < {self.velocity_threshold})"

        plt.title(
            f'{meta2["title"]} zu {meta1["title"]} \n{correlation_text}',
            fontsize=20,
            pad=25,
            color=self.kit_dark_blue,
            weight='bold'
        )
        plt.xlabel(
            meta1["axis"],
            fontsize=16,
            color=self.kit_dark_blue,
            labelpad=15
        )
        plt.ylabel(
            meta2["axis"],
            fontsize=16,
            color=self.kit_dark_blue,
            labelpad=15,
            rotation=0
        )

        # Achsen und Gitter anpassen
        plt.grid(True, linestyle='--', alpha=0.6, color='0.7')
        plt.tick_params(
            axis='both',
            which='both',
            colors=self.kit_dark_blue,
            labelsize=14,
            length=5,
            width=1.5
        )

        # Hintergrund und Ränder
        plt.gca().set_facecolor('white')
        for spine in plt.gca().spines.values():
            spine.set_edgecolor(self.kit_dark_blue)
            spine.set_linewidth(1.5)

        # Dichteplot nur bei geeigneten Daten hinzufügen
        if self._should_add_density(df_filtered[column1], df_filtered[column2]):
            try:
                sns.kdeplot(
                    x=df_filtered[column1],
                    y=df_filtered[column2],
                    cmap="Blues",
                    alpha=0.2,
                    thresh=0.05,
                    levels=5,
                    fill=True,
                    linewidths=0
                )
            except Exception as e:
                print(f"Konnte Dichteplot nicht erstellen: {str(e)}")

        # Speichern mit hochauflösendem Format
        if output_filename is None:
            output_filename = f"correlation_{column1}_vs_{column2}.png"
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(
            output_path,
            dpi=600,
            bbox_inches='tight',
            facecolor='white',
            transparent=False
        )
        plt.close()
        print(f"Gespeichert: {output_path}")
        return output_path

    def calculate_terms(self, df):
        """Berechnet die Term_{axis} Spalten"""
        for axis in ['x', 'y', 'z', 'sp']:
            df[f'Term_{axis}'] = df[f'f_{axis}_sim'] * df['materialremoved_sim'] / (df[f'v_{axis}'] + 1e-3)
        return df

    def generate_all_plots(self, csv_paths):
        """Generiert alle Korrelationsplots für die gegebenen Daten"""
        # Daten laden und vorbereiten
        df = self.load_csv_files(csv_paths)
        df = self.calculate_terms(df)
        # Alle Kombinationen plotten
        plot_paths = []
        for axis in ['x', 'y', 'z', 'sp']:
            column1 = f"curr_{axis}"
            # Standard-Kombinationen
            standard_cols = [f'v_{axis}']
            for col in standard_cols:
                if col in df.columns:  # Überprüfen ob Spalte existiert
                    filename = f"corr_{column1}_vs_{col}.png"
                    try:
                        path = self.plot_correlation(df, column1, col, filename)
                        if path is not None:  # Nur hinzufügen wenn Plot erfolgreich erstellt wurde
                            plot_paths.append(path)
                    except Exception as e:
                        print(f"Fehler beim Erstellen von {filename}: {str(e)}")
                        continue
        return plot_paths

# Beispielaufruf
if __name__ == "__main__":
    # **Schwellwert für |v| setzen (z. B. 1 mm/s)**
    plotter = CorrelationPlotter(velocity_threshold=10.0)  # Hier den gewünschten Schwellwert eintragen

    # Pfade zu den CSV-Dateien
    csv_paths = ['DMC60H_S235JR_Plate_Normal_1.csv', 'DMC60H_S235JR_Plate_Normal_2.csv']

    # Alle Plots generieren
    print("Starte Plot-Generierung...")
    try:
        generated_plots = plotter.generate_all_plots(csv_paths)
        print(f"\nErfolgreich erstellte Plots ({len(generated_plots)}):")
        for path in generated_plots:
            print(f"- {os.path.basename(path)}")
    except Exception as e:
        print(f"Fehler bei der Plot-Generierung: {str(e)}")
