import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List
from dataclasses import dataclass

from scipy.ndimage import uniform_filter1d


@dataclass
class MillingSegment:
    """Datenstruktur für ein Fräs-Segment."""
    start_idx: int
    end_idx: int
    z_position: float
    feed_velocity: float
    positions: np.ndarray  # [N, 3] - XYZ Positionen
    cut_depths: np.ndarray  # [N] - Schnitttiefe für jeden Punkt


def create_circle_mask(tool_radius_voxels: int) -> np.ndarray:
    """Erstellt eine kreisförmige Maske für das Werkzeug."""
    size = 2 * tool_radius_voxels + 1
    y_grid, x_grid = np.meshgrid(
        np.arange(-tool_radius_voxels, tool_radius_voxels + 1),
        np.arange(-tool_radius_voxels, tool_radius_voxels + 1),
        indexing='ij'
    )
    circle_mask = (x_grid ** 2 + y_grid ** 2) <= (tool_radius_voxels ** 2)
    return circle_mask


def remove_material_at_position_old(workpiece: np.ndarray,
                                    pos_x: float, pos_y: float, pos_z: float,
                                    cut_depth: float,
                                    circle_mask: np.ndarray,
                                    part_position: np.ndarray,
                                    voxel_size: float,
                                    tool_radius_voxels: int) -> Tuple[np.ndarray, float]:
    """
    Entfernt Material an einer Werkzeugposition.
    Basiert auf der static_mill_path Logik.
    """
    nx, ny, nz = workpiece.shape

    # Voxel-Koordinaten berechnen
    vx = int(round((pos_x - part_position[0]) / voxel_size))
    vy = int(round((pos_y - part_position[1]) / voxel_size))
    vz = int(round((pos_z - part_position[2]) / voxel_size))

    # Schnitttiefe in Voxel
    cut_voxels = max(1, int(round(cut_depth / voxel_size)))

    # Grenzen bestimmen
    x_min = max(0, vx - tool_radius_voxels)
    x_max = min(nx, vx + tool_radius_voxels + 1)
    y_min = max(0, vy - tool_radius_voxels)
    y_max = min(ny, vy + tool_radius_voxels + 1)
    z_start = max(0, vz)
    z_end = min(nz, vz + cut_voxels)

    # Prüfen ob Position im gültigen Bereich
    if x_min >= x_max or y_min >= y_max or z_start >= z_end:
        return workpiece, 0.0

    # Maske für aktuellen Bereich anpassen
    width = x_max - x_min
    depth = y_max - y_min

    # Offset für die Kreismaske berechnen
    mask_x_start = max(0, tool_radius_voxels - (vx - x_min))
    mask_y_start = max(0, tool_radius_voxels - (vy - y_min))
    mask_x_end = min(circle_mask.shape[1], mask_x_start + width)
    mask_y_end = min(circle_mask.shape[0], mask_y_start + depth)

    # Lokale Maske extrahieren
    local_mask = circle_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

    # Sicherstellen dass Maske und Region gleiche Größe haben
    actual_height, actual_width = local_mask.shape
    if actual_height != depth or actual_width != width:
        # Größe anpassen falls nötig
        depth = min(depth, actual_height)
        width = min(width, actual_width)
        x_max = x_min + width
        y_max = y_min + depth
        local_mask = local_mask[:depth, :width]

    # Material über Z-Ebenen entfernen
    removed_material = 0.0
    workpiece_copy = workpiece.copy()

    for z in range(z_start, z_end):
        if z >= nz:
            break

        # Aktueller Bereich
        current_region = workpiece_copy[x_min:x_max, y_min:y_max, z]

        # Material nur entfernen wo Maske aktiv ist UND Material vorhanden
        material_to_remove = current_region * local_mask.T
        removed_material += np.sum(material_to_remove)

        # Material entfernen
        workpiece_copy[x_min:x_max, y_min:y_max, z] =  current_region * (~local_mask).T

    return workpiece_copy, removed_material


class SimpleCNCMRRSimulation:
    def __init__(self, part_position: list, part_dimension: list, tool_radius: float, sampling_frequency: float,):
        self.part_position = np.array(part_position)
        self.part_width, self.part_depth, self.part_height, self.voxel_size = part_dimension
        self.tool_radius = tool_radius
        self.sampling_frequency = sampling_frequency
        # Voxel-Grid Dimensionen
        self.nx = int(self.part_width / self.voxel_size)
        self.ny = int(self.part_depth / self.voxel_size)
        self.nz = int(self.part_height / self.voxel_size)
        # Werkzeugradius in Voxel-Einheiten
        self.tool_radius_voxels = int(np.ceil(self.tool_radius / self.voxel_size))
        # Werkstück initialisieren (1.0 = Material vorhanden)
        self.material_grid = np.ones((self.nx, self.ny, self.nz), dtype=np.float32)
        # Kreismaske für Werkzeug
        self.circle_mask = self.create_circle_mask(self.tool_radius_voxels)
        print(f"Voxel-Grid: {self.nx}×{self.ny}×{self.nz} = {self.nx * self.ny * self.nz:,} Voxel")
        print(f"Werkzeugradius: {self.tool_radius}mm ({self.tool_radius_voxels} Voxel)")
        print(f"Kreismaske Größe: {self.circle_mask.shape}")

    def create_circle_mask(self, radius: int) -> np.ndarray:
        """Erstellt eine kreisförmige Maske mit gegebenem Radius."""
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = x**2 + y**2 <= radius**2
        return mask.astype(float)

    def process_all_positions(self, df: pd.DataFrame) -> Tuple[np.ndarray, float, List[float]]:
        total_removed = 0.0
        current_workpiece = self.material_grid.copy()
        removed_voxels_per_point = []

        positions = df[['pos_x', 'pos_y', 'pos_z']].values
        cut_depths = df['a_p'].values if 'a_p' in df.columns else np.ones(len(df)) * 1.0

        for i, (position, cut_depth) in enumerate(zip(positions, cut_depths)):
            pos_x, pos_y, pos_z = position
            current_workpiece, removed = self.remove_material_at_position(
                current_workpiece,
                pos_x, pos_y, pos_z, cut_depth,
                self.circle_mask,
                self.part_position,
                self.voxel_size,
                self.tool_radius_voxels
            )
            removed_voxels_per_point.append(removed)
            total_removed += removed
            status = i / len(positions) * 100
            print(f'Status: {status:.2f}% Removed: {removed:.2f}')
        return current_workpiece, total_removed, removed_voxels_per_point

    @staticmethod
    def remove_material_at_position(workpiece: np.ndarray,
                                    pos_x: float, pos_y: float, pos_z: float,
                                    cut_depth: float,
                                    circle_mask: np.ndarray,
                                    part_position: np.ndarray,
                                    voxel_size: float,
                                    tool_radius_voxels: int) -> Tuple[np.ndarray, float]:
        nx, ny, nz = workpiece.shape
        # Voxel-Koordinaten berechnen
        vx = int(round((pos_x - part_position[0]) / voxel_size))
        vy = int(round((pos_y - part_position[1]) / voxel_size))
        vz = int(round((pos_z - part_position[2]) / voxel_size))

        # Schnitttiefe in Voxel
        cut_voxels = max(1, int(round(cut_depth / voxel_size)))

        # Grenzen bestimmen
        x_min = max(0, vx - tool_radius_voxels)
        x_max = min(nx, vx + tool_radius_voxels + 1)
        y_min = max(0, vy - tool_radius_voxels)
        y_max = min(ny, vy + tool_radius_voxels + 1)
        z_start = max(0, vz)
        z_end = min(nz, vz + cut_voxels)

        # Prüfen ob Position im gültigen Bereich
        if x_min >= x_max or y_min >= y_max or z_start >= z_end:
            return workpiece, 0.0

        width = x_max - x_min
        depth = y_max - y_min
        mask_x_start = max(0, tool_radius_voxels - (vx - x_min))
        mask_y_start = max(0, tool_radius_voxels - (vy - y_min))
        mask_x_end = min(circle_mask.shape[1], mask_x_start + width)
        mask_y_end = min(circle_mask.shape[0], mask_y_start + depth)

        # Lokale Maske extrahieren
        local_mask = circle_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

        # Größe anpassen falls nötig
        if local_mask.shape[0] != depth or local_mask.shape[1] != width:
            local_mask = local_mask[:depth, :width]

        # 3D-Maske erstellen
        mask_3d = np.tile(local_mask[:, :, np.newaxis], (1, 1, z_end - z_start))

        # Sicherstellen, dass die Dimensionen übereinstimmen
        mask_3d = mask_3d.T  # Transponieren, um die Dimensionen anzupassen
        mask_3d = np.transpose(mask_3d, (1, 2, 0))  # Dimensionen anpassen

        # Material über Z-Ebenen entfernen
        workpiece_region = workpiece[x_min:x_max, y_min:y_max, z_start:z_end]
        material_to_remove = workpiece_region * mask_3d
        removed_material = np.sum(material_to_remove)
        workpiece[x_min:x_max, y_min:y_max, z_start:z_end] = (
                workpiece_region - material_to_remove
        )

        return workpiece, removed_material

    def simulate_mrr(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hauptsimulation der MRR-Berechnung.
        Args:
            df: DataFrame mit Pfaddaten
            sampling_frequency: Abtastfrequenz in Hz
        Returns:
            (times, mrr_values)
        """
        print("Starte MRR-Simulation...")
        # Schritt 1: Alle Positionen auf einmal verarbeiten
        current_workpiece, total_removed, removed_voxels = self.process_all_positions(df)

        # Schritt 2: MRR-Berechnung
        print("Berechne MRR...")
        times, mrr_values = self.calculate_mrr(removed_voxels)
        print("Simulation abgeschlossen!")
        return times, mrr_values

    def calculate_mrr(self, removed_voxels: List[float], filter_window: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        voxel_volume = self.voxel_size ** 3
        removed_voxels_array = np.array(removed_voxels)

        # Mittelwertfilter anwenden
        if len(removed_voxels_array) >= filter_window:
            filtered_voxels = uniform_filter1d(removed_voxels_array, size=filter_window, mode="reflect")
        else:
            filtered_voxels = uniform_filter1d(removed_voxels_array, size=len(removed_voxels_array), mode="reflect")

        # Zeitpunkte berechnen
        time_per_point = 1.0 / self.sampling_frequency
        total_duration = len(filtered_voxels) / self.sampling_frequency
        times = np.linspace(0, total_duration, len(filtered_voxels))

        # MRR berechnen
        mrr_values = filtered_voxels  * voxel_volume / time_per_point
        return times, mrr_values

    def plot_results(self, times: np.ndarray, mrr_values: np.ndarray):
        """Plot the simulation results."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # MRR über Zeit
        ax.plot(times, mrr_values, 'b-', linewidth=1.5, label='MRR')
        ax.set_xlabel('Zeit (s)')
        ax.set_ylabel('MRR (mm³/s)')
        ax.set_title('Material Removal Rate über Zeit')
        ax.grid(True, alpha=0.3)

        # Statistiken
        mean_mrr = np.mean(mrr_values)
        max_mrr = np.max(mrr_values)
        ax.axhline(mean_mrr, color='g', linestyle='--', alpha=0.7,
                   label=f'Mittelwert: {mean_mrr:.2f} mm³/s')
        ax.legend()

        plt.tight_layout()
        plt.show()

        # Erweiterte Statistiken
        print(f"\n=== MRR Statistiken ===")
        print(f"Maximum MRR: {max_mrr:.3f} mm³/s")
        print(f"Mittlere MRR: {mean_mrr:.3f} mm³/s")
        print(f"Standardabweichung: {np.std(mrr_values):.3f} mm³/s")
        print(f"Gesamtvolumen entfernt: {np.trapz(mrr_values, times):.1f} mm³")
        print(f"Simulationszeit: {times[-1]:.2f} s")

# Beispiel-Nutzung
if __name__ == "__main__":
    # Parameter
    part_position = [-35.64, 175.0, 354.94]
    part_dimension = [75.0, 75.0 * 2, 50.0, 0.5]  # Größere Voxel für bessere Performance
    tool_radius = 5.0

    # Testdaten laden
    try:
        sample_data_path = '..\\..\\DataSetsV3/DataMerged/S235JR_Plate_Normal.csv'
        sample_df = pd.read_csv(sample_data_path)

        # Schnitttiefe hinzufügen falls nicht vorhanden
        if 'a_p' not in sample_df.columns:
            sample_df['a_p'] = 1.0  # Default 1mm Schnitttiefe

        # Daten reduzieren für Testzwecke
        #n = min(500, len(sample_df))  # Nur 500 Punkte für ersten Test
        #sample_df = sample_df[:n]

        print(f"Lade {len(sample_df)} Datenpunkte...")
        print(f"Verfügbare Spalten: {list(sample_df.columns)}")

    except FileNotFoundError:
        print("Testdaten nicht gefunden. Erstelle Beispieldaten...")

        # Spiral-Pfad erstellen
        n_points = 300
        t = np.linspace(0, 10, n_points)
        radius = np.linspace(15, 3, n_points)
        angle = t * 3 * np.pi

        sample_df = pd.DataFrame({
            'pos_x': part_position[0] + radius * np.cos(angle),
            'pos_y': part_position[1] + radius * np.sin(angle),
            'pos_z': part_position[2] + 15 - t * 1.5,  # Nach unten fräsen
            'v_x': np.gradient(radius * np.cos(angle)),
            'v_y': np.gradient(radius * np.sin(angle)),
            'v_z': np.gradient(part_position[2] + 15 - t * 1.5),
            'a_p': 0.8 + 0.4 * np.sin(t * 2)  # Variable Schnitttiefe 0.4-1.2mm
        })

        print(f"Erstellt {len(sample_df)} Beispiel-Datenpunkte")

    # Simulation ausführen
    print("\nStarte Simulation...")
    simulator = SimpleCNCMRRSimulation(part_position, part_dimension, tool_radius)

    times, mrr_values, segments = simulator.simulate_mrr(sample_df)

    # Ergebnisse plotten
    simulator.plot_results(times, mrr_values, segments, sample_df)

    sample_df['MRR'] = mrr_values

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    if 'time' in sample_df.columns:
        time = sample_df['time']
    else:
        time = sample_df.index * (1 / 50)
    label = 'MRR'
    ax1.plot(time, sample_df[label], label=label, color='tab:green')
    ax1.set_xlabel('Time in s')

    ax1.legend(loc='upper left')


    # Zweite y-Achse für curr_x
    ylabel = 'curr_x'
    ax2 = ax1.twinx()
    ax2.plot(time, sample_df[ylabel], label=ylabel, color='tab:red', linestyle='--', alpha=0.8)
    ax2.set_ylabel(ylabel)
    ax2.legend(loc='upper right')


    plt.show()