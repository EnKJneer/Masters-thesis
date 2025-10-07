import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import Helper.handling_data as hdata
from matplotlib.colors import LinearSegmentedColormap

class CorrelationPlotter:
    """Erstellt Korrelationsplots mit KIT-Designrichtlinien, Ableitungspfeilen und standardisierten Achsenbezeichnungen"""
    def __init__(self, output_dir="Plot_Correlation", velocity_threshold=None, derivative_arrow_step=10, force_threshold = 0):
        self.output_dir = output_dir
        self.velocity_threshold = velocity_threshold  # Schwellwert für |v|
        self.force_threshold = force_threshold
        self.derivative_arrow_step = derivative_arrow_step  # Jeder n-te Punkt für Ableitungspfeile (Default: 10)
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
        """Entfernt Outlier aus dem DataFrame für die gegebenen Spalten."""
        df_clean = df.copy()
        for col in [column1, column2]:
            if col in df_clean.columns:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                lower_bound = mean - threshold_std * std
                upper_bound = mean + threshold_std * std
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        return df_clean

    def _filter_by_velocity_threshold(self, df, column1, column2):
        """Filtert Datenpunkte basierend auf dem Schwellwert für |v|."""
        if self.velocity_threshold is None:
            return df
        v_columns = [col for col in [column1, column2] if col.startswith('v_')]
        if not v_columns:
            return df
        for v_col in v_columns:
            df = df[abs(df[v_col]) < self.velocity_threshold]
        return df

    def _filter_by_force_threshold(self, df):
        """Filtert Datenpunkte basierend auf dem Schwellwert für |v|."""

        df = df[df['f_x_sim'] > self.force_threshold]
        return df

    def _filter_zero_values(self, df, column1, column2):
        """Filtert Datenpunkte, bei denen einer der Werte 0 ist."""
        mask = (df[column1] != 0) & (df[column2] != 0)
        mask &= (abs(df[column1]) > 1e-6) & (abs(df[column2]) > 1e-6)
        return df[mask]

    def _should_add_density(self, x_data, y_data):
        """Entscheidet, ob ein Dichteplot hinzugefügt werden soll."""
        if len(x_data) < 100:
            return False
        x_var = np.var(x_data)
        y_var = np.var(y_data)
        return x_var > 1e-6 and y_var > 1e-6 and len(x_data) > 100

    def _add_derivative_arrows(self, ax, x_data, y_data, step=10, scale=1.0, color=None):
        """
        Zeichnet Pfeile für die lokale Ableitung (Steigung) ein.
        Args:
            ax: Matplotlib-Achsenobjekt
            x_data: x-Werte der Daten
            y_data: y-Werte der Daten
            step: Jeder n-te Punkt wird berücksichtigt (Default: 10)
            scale: Skalierungsfaktor für die Pfeillänge
            color: Farbe der Pfeile (Default: KIT-Rot)
        """
        if color is None:
            color = self.kit_red

        # Berechne die Ableitung (Steigung) zwischen aufeinanderfolgenden Punkten
        dx = np.gradient(x_data)
        dy = np.gradient(y_data)
        slopes = dy / (dx + 1e-10)  # Vermeide Division durch 0

        # Zeichne Pfeile für jeden n-ten Punkt
        for i in range(0, len(x_data), step):
            if i >= len(x_data) - 1:
                continue  # Überspringe den letzten Punkt (kein Nachfolger)

            x, y = x_data[i], y_data[i]
            dx_local = dx[i]
            dy_local = dy[i]

            # Skalierung der Pfeillänge (vermeide zu lange Pfeile)
            arrow_length = np.sqrt(dx_local**2 + dy_local**2)
            if arrow_length > 0:
                dx_local /= (arrow_length / scale)
                dy_local /= (arrow_length / scale)

            # Zeichne den Pfeil
            ax.arrow(
                x, y,
                dx_local, dy_local,
                head_width=0.005 * (max(x_data) - min(x_data)),
                head_length=0.01 * (max(y_data) - min(y_data)),
                fc=color, ec=color,
                alpha=0.7,
                linewidth=1.5,
                length_includes_head=True
            )

    def get_column_metadata(self, column_name):
        """Gibt die Metadaten für eine Spalte zurück oder Standardwerte."""
        if column_name in self.column_metadata:
            return self.column_metadata[column_name]
        return {
            'axis': column_name.replace('_', ' ').title(),
            'title': column_name.replace('_', ' ').title(),
            'unit': ''
        }

    def load_csv_files(self, csv_paths):
        """Lädt mehrere CSV-Dateien in ein DataFrame."""
        dfs = []
        for path in csv_paths:
            file_path = os.path.join(hdata.DataClass_ST_Plate_Notch.folder, path)
            df = pd.read_csv(file_path)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def plot_correlation(self, df, column1, column2, output_filename=None):
        """
        Erstellt einen Korrelationsplot mit KIT-Design, Ableitungspfeilen und speichert ihn.
        """
        # Daten filtern
        df_clean = self._remove_outliers(df, column1, column2)
        df_filtered = self._filter_zero_values(df_clean, column1, column2)
        df_filtered = self._filter_by_velocity_threshold(df_filtered, column1, column2)
        df_filtered = self._filter_by_force_threshold(df_filtered)

        if len(df_filtered) < 10:
            print(f"Zu wenige Datenpunkte nach Filterung für {column1} vs {column2} ({len(df_filtered)} Punkte)")
            return None

        # Metadaten und Korrelation
        meta1 = self.get_column_metadata(column1)
        meta2 = self.get_column_metadata(column2)
        correlation = df_filtered[[column1, column2]].corr().iloc[0, 1]

        # Plot erstellen
        plt.figure(figsize=(14, 10))
        sns.set_style("whitegrid")

        # Scatterplot
        sns.scatterplot(
            data=df_filtered,
            x=column1,
            y=column2,
            color=self.kit_blue,
            alpha=0.7,
            edgecolor='white',
            linewidth=0.5,
            s=80
        )

        # **Ableitungspfeile hinzufügen**
        x_data = df_filtered[column1].values
        y_data = df_filtered[column2].values
        self._add_derivative_arrows(plt.gca(), x_data, y_data, step=self.derivative_arrow_step, color=self.kit_red)

        # Titel und Achsen
        correlation_text = f"Korrelation: {correlation:.3f}"
        if self.velocity_threshold is not None:
            correlation_text += f" (|v| < {self.velocity_threshold})"
        if self.derivative_arrow_step > 0:
            correlation_text += f" | Ableitungspfeile (1/{self.derivative_arrow_step})"

        plt.title(
            f'{meta2["title"]} zu {meta1["title"]}\n{correlation_text}',
            fontsize=20,
            pad=25,
            color=self.kit_dark_blue,
            weight='bold'
        )
        plt.xlabel(meta1["axis"], fontsize=16, color=self.kit_dark_blue, labelpad=15)
        plt.ylabel(meta2["axis"], fontsize=16, color=self.kit_dark_blue, labelpad=15, rotation=0)

        # Gitter und Stile
        plt.grid(True, linestyle='--', alpha=0.6, color='0.7')
        plt.tick_params(axis='both', colors=self.kit_dark_blue, labelsize=14, length=5, width=1.5)
        plt.gca().set_facecolor('white')
        for spine in plt.gca().spines.values():
            spine.set_edgecolor(self.kit_dark_blue)
            spine.set_linewidth(1.5)

        # Dichteplot (optional)
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

        # Speichern
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

    def generate_all_plots(self, csv_paths):
        """Generiert alle Korrelationsplots für die gegebenen Daten."""
        df = self.load_csv_files(csv_paths)
        plot_paths = []
        for axis in ['x', 'y']:
            column1 = f"curr_{axis}"
            standard_cols = [f'f_{axis}_sim']
            for col in standard_cols:
                if col in df.columns:
                    filename = f"corr_{column1}_vs_{col}.png"
                    try:
                        path = self.plot_correlation(df, column1, col, filename)
                        if path is not None:
                            plot_paths.append(path)
                    except Exception as e:
                        print(f"Fehler beim Erstellen von {filename}: {str(e)}")
        return plot_paths

# Beispielaufruf
if __name__ == "__main__":
    # **Schwellwert für |v| und Ableitungspfeile setzen**
    plotter = CorrelationPlotter(
        velocity_threshold=10.0,  # Schwellwert für |v|
        derivative_arrow_step=100   # Jeder 10. Punkt bekommt einen Ableitungspfeil
    )

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
