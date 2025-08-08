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
    def __init__(self, part_position: list, part_dimension: list, tool_radius: float):
        """
        Einfache CNC MRR Simulation ohne JAX.

        Args:
            part_position: [x, y, z] Position des Werkstücks
            part_dimension: [width, depth, height, voxel_size]
            tool_radius: Werkzeugradius in mm
        """
        self.part_position = np.array(part_position)
        self.part_width, self.part_depth, self.part_height, self.voxel_size = part_dimension
        self.tool_radius = tool_radius

        # Voxel-Grid Dimensionen
        self.nx = int(self.part_width / self.voxel_size)
        self.ny = int(self.part_depth / self.voxel_size)
        self.nz = int(self.part_height / self.voxel_size)

        # Werkzeugradius in Voxel-Einheiten
        self.tool_radius_voxels = int(np.ceil(self.tool_radius / self.voxel_size))

        # Segmentierungs-Parameter
        self.z_tolerance = 0.1  # mm
        self.velocity_tolerance = 0.6

        # Werkstück initialisieren (1.0 = Material vorhanden)
        self.material_grid = np.ones((self.nx, self.ny, self.nz), dtype=np.float32)

        # Kreismaske für Werkzeug
        self.circle_mask = create_circle_mask(self.tool_radius_voxels)

        print(f"Voxel-Grid: {self.nx}×{self.ny}×{self.nz} = {self.nx * self.ny * self.nz:,} Voxel")
        print(f"Werkzeugradius: {self.tool_radius}mm ({self.tool_radius_voxels} Voxel)")
        print(f"Kreismaske Größe: {self.circle_mask.shape}")

    def is_tool_fully_inside(self, pos_x: float, pos_y: float, pos_z: float, cut_depth: float) -> bool:
        """Prüft, ob das Werkzeug vollständig im Werkstück liegt (X/Y mit Radius, Z mit Schnitttiefe)."""
        x_min = pos_x - self.tool_radius
        x_max = pos_x + self.tool_radius
        y_min = pos_y - self.tool_radius
        y_max = pos_y + self.tool_radius
        z_min = pos_z - cut_depth/2
        z_max = pos_z + cut_depth/2  # Werkzeugspitze

        part_x_min = self.part_position[0]
        part_x_max = self.part_position[0] + self.part_width/2
        part_y_min = self.part_position[1]
        part_y_max = self.part_position[1] + self.part_depth/2
        part_z_min = self.part_position[2]
        part_z_max = self.part_position[2] + self.part_height/2

        return (
                x_min >= part_x_min and x_max <= part_x_max and
                y_min >= part_y_min and y_max <= part_y_max and
                z_min >= part_z_min and z_max <= part_z_max
        )

    def segment_path_data(self, df: pd.DataFrame) -> List[MillingSegment]:
        """
        Segmentiert die Pfaddaten in drei Phasen:
        - Tool vollständig außerhalb: ein einzelnes Segment
        - Eintrittsphase: jeder Punkt einzeln
        - Tool vollständig im Material: heuristische Segmentierung wie bisher
        """
        print("Segmentiere Pfaddaten...")

        positions = df[['pos_x', 'pos_y', 'pos_z']].values
        cut_depths = df['a_p'].values if 'a_p' in df.columns else np.ones(len(df)) * 1.0
        velocities_xy = np.sqrt(df['v_x'].values ** 2 + df['v_y'].values ** 2)

        segments = []
        i = 0
        n = len(positions)
        # ToDo: Nur ein segment erstellen
        while i < n:
            z = positions[i, 2]
            a_p = cut_depths[i]
            in_material = self.is_tool_fully_inside(positions[i, 0], positions[i, 1], positions[i, 2], a_p)

            # Phase 1: vollständig außerhalb
            if not in_material:
                start = i
                while i < n and not self.is_tool_fully_inside(positions[i, 0], positions[i, 1], positions[i, 2], a_p):
                    i += 1
                segment = MillingSegment(
                    start_idx=start,
                    end_idx=i,
                    z_position=positions[start, 2],
                    feed_velocity=velocities_xy[start],
                    positions=positions[start:i].copy(),
                    cut_depths=cut_depths[start:i].copy()
                )
                segments.append(segment)

            # Phase 2: Eintrittsphase (nicht vollständig im Material)
            elif not self.is_tool_fully_inside(positions[i, 0], positions[i, 1], positions[i, 2], a_p) and i + 1 < n and self.is_tool_fully_inside(positions[i, 0], positions[i, 1], positions[i, 2], a_p):
                segment = MillingSegment(
                    start_idx=i,
                    end_idx=i + 1,
                    z_position=z,
                    feed_velocity=velocities_xy[i],
                    positions=positions[i:i + 1].copy(),
                    cut_depths=cut_depths[i:i + 1].copy()
                )
                segments.append(segment)
                i += 1

            # Phase 3: Werkzeug vollständig im Material → Standard-Segmentierung
            else:
                start = i
                i += 1
                while i < n:
                    z_change = abs(positions[i, 2] - positions[start, 2]) > self.z_tolerance
                    if velocities_xy[start] > 0.001:
                        velocity_change = abs(velocities_xy[i] - velocities_xy[start]) / velocities_xy[
                            start] > self.velocity_tolerance
                    else:
                        velocity_change = velocities_xy[i] > 0.001

                    if z_change or velocity_change or not self.is_tool_fully_inside(positions[i, 0], positions[i, 1], positions[i, 2], a_p):
                        break
                    i += 1

                segment = MillingSegment(
                    start_idx=start,
                    end_idx=i,
                    z_position=positions[start, 2],
                    feed_velocity=velocities_xy[start],
                    positions=positions[start:i].copy(),
                    cut_depths=cut_depths[start:i].copy()
                )
                segments.append(segment)

        print(f"Erstellt {len(segments)} Segmente:")
        for i, seg in enumerate(segments[:5]):
            print(
                f"  Segment {i + 1}: {len(seg.positions)} Punkte, Z={seg.z_position:.2f}mm, v={seg.feed_velocity:.2f}mm/s")
        if len(segments) > 5:
            print(f"  ... und {len(segments) - 5} weitere Segmente")

        return segments

    def process_segment(self, segment: MillingSegment,
                        workpiece: np.ndarray) -> Tuple[np.ndarray, float, List[float]]:
        total_removed = 0.0
        current_workpiece = workpiece.copy()
        removed_voxels_per_point = np.zeros(len(segment.positions))  # Nutzung von NumPy Array für bessere Performance

        for i, (position, cut_depth) in enumerate(zip(segment.positions, segment.cut_depths)):
            pos_x, pos_y, pos_z = position
            current_workpiece, removed = self.remove_material_at_position(
                current_workpiece,
                pos_x, pos_y, pos_z, cut_depth,
                self.circle_mask,
                self.part_position,
                self.voxel_size,
                self.tool_radius_voxels
            )
            removed_voxels_per_point[i] = removed
            total_removed += removed
            status = i / len(segment.positions) * 100
            print(f'Status: {status:.2f}% Removed: {removed:.2f}')

        return current_workpiece, total_removed, removed_voxels_per_point.tolist()

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

    def process_segment_old(self, segment: MillingSegment,
                        workpiece: np.ndarray) -> Tuple[np.ndarray, float, List[float]]:
        """
        Gibt zusätzlich Voxel-Abtragsliste pro Punkt zurück.
        """
        total_removed = 0.0
        current_workpiece = workpiece.copy()
        removed_voxels_per_point = []

        for i, (position, cut_depth) in enumerate(zip(segment.positions, segment.cut_depths)):

            #if i % 2 == 0:
            pos_x, pos_y, pos_z = position

            current_workpiece, removed = remove_material_at_position_old(
                current_workpiece,
                pos_x, pos_y, pos_z, cut_depth,
                self.circle_mask,
                self.part_position,
                self.voxel_size,
                self.tool_radius_voxels
            )

            removed_voxels_per_point.append(removed)
            total_removed += removed
            status = i / len(segment.positions) * 100
            print(f'Status: {status:.2f}% Removed: {removed:.2f}')
            #else:
            #    removed_voxels_per_point.append(0)
        return current_workpiece, total_removed, removed_voxels_per_point

    def simulate_mrr(self, df: pd.DataFrame,
                     sampling_frequency: float = 50.0) -> Tuple[np.ndarray, np.ndarray, List[MillingSegment]]:
        """
        Hauptsimulation der MRR-Berechnung.

        Args:
            df: DataFrame mit Pfaddaten
            sampling_frequency: Abtastfrequenz in Hz

        Returns:
            (times, mrr_values, segments)
        """
        print("Starte MRR-Simulation...")

        # Schritt 1: Segmentierung
        segments = self.segment_path_data(df)

        # Schritt 2: Segment-Verarbeitung
        print("Verarbeite Segmente...")
        removed_materials = []
        current_workpiece = self.material_grid

        removed_voxels_all_points = []
        for seg_idx, segment in enumerate(segments):
            print(f"Verarbeite Segment {seg_idx + 1}/{len(segments)}: {len(segment.positions)} Punkte")

            current_workpiece, removed_material, removed_per_point = self.process_segment(segment, current_workpiece)
            removed_materials.append(removed_material)
            removed_voxels_all_points.append(removed_per_point)

            print(f"  → {removed_material:.1f} Voxel entfernt")

        # Schritt 3: MRR-Berechnung
        print("Berechne MRR aus Segmenten...")
        times, mrr_values = self.calculate_mrr_from_segments(
            segments, removed_voxels_all_points, sampling_frequency
        )

        print("Simulation abgeschlossen!")
        return times, mrr_values, segments

    def exponential_smoothing(self, data: np.ndarray, alpha: float) -> np.ndarray:
        """Apply exponential smoothing to the data."""
        smoothed_data = np.zeros_like(data)
        if len(data) > 0:
            smoothed_data[0] = data[0]  # Initial value
            for i in range(1, len(data)):
                smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]
        return smoothed_data

    def calculate_mrr_from_segments(self, segments: List[MillingSegment],
                                    removed_voxels_all_points: List[List[float]],
                                    sampling_frequency: float,
                                    filter_window: int = 40) -> Tuple[np.ndarray, np.ndarray]:

        voxel_volume = self.voxel_size ** 3
        all_times = []
        all_mrr_values = []
        current_time = 0.0

        for segment, removed_voxels_per_point in zip(segments, removed_voxels_all_points):
            removed_voxels_per_point = np.array(removed_voxels_per_point)

            # Mittelwertfilter anwenden, am Rand mit kürzerem Fenster
            if len(removed_voxels_per_point) >= filter_window:
                filtered_voxels = uniform_filter1d(
                    removed_voxels_per_point, size=filter_window, mode="reflect"
                )
            else:
                filtered_voxels = uniform_filter1d(
                    removed_voxels_per_point, size=len(removed_voxels_per_point), mode="reflect"
                )

            segment_duration = len(filtered_voxels) / sampling_frequency
            time_per_point = 1.0 / sampling_frequency

            segment_times = np.linspace(current_time,
                                        current_time + segment_duration,
                                        len(filtered_voxels))

            mrr_values = filtered_voxels * voxel_volume / time_per_point

            all_times.extend(segment_times)
            all_mrr_values.extend(mrr_values)

            current_time += segment_duration

        return np.array(all_times), np.array(all_mrr_values)


    def plot_results(self, times: np.ndarray, mrr_values: np.ndarray,
                     segments: List[MillingSegment], df: pd.DataFrame = None):
        """Plottet die Ergebnisse der Simulation."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # MRR über Zeit
        axes[0].plot(times, mrr_values, 'b-', linewidth=1.5, label='MRR')
        axes[0].set_xlabel('Zeit (s)')
        axes[0].set_ylabel('MRR (mm³/s)')
        axes[0].set_title('Material Removal Rate über Zeit')
        axes[0].grid(True, alpha=0.3)

        # Segment-Grenzen markieren
        current_time = 0
        for i, segment in enumerate(segments[:10]):  # Nur erste 10 Segmente markieren
            segment_duration = len(segment.positions) / 1000.0
            axes[0].axvline(current_time, color='r', linestyle='--', alpha=0.3)
            current_time += segment_duration

        # Statistiken
        mean_mrr = np.mean(mrr_values)
        max_mrr = np.max(mrr_values)
        axes[0].axhline(mean_mrr, color='g', linestyle='--', alpha=0.7,
                        label=f'Mittelwert: {mean_mrr:.2f} mm³/s')
        axes[0].legend()

        # Segment-Analyse
        segment_mrrs = []
        current_idx = 0
        for segment in segments:
            seg_len = len(segment.positions)
            if current_idx + seg_len <= len(mrr_values):
                segment_mrrs.append(np.mean(mrr_values[current_idx:current_idx + seg_len]))
            current_idx += seg_len

        if segment_mrrs:
            axes[1].bar(range(len(segment_mrrs)), segment_mrrs, alpha=0.7)
            axes[1].set_xlabel('Segment')
            axes[1].set_ylabel('Mittlere MRR (mm³/s)')
            axes[1].set_title('MRR pro Segment')
            axes[1].grid(True, alpha=0.3)

        # Segment-Parameter
        if segments:
            z_positions = [seg.z_position for seg in segments]
            feed_velocities = [seg.feed_velocity for seg in segments]

            ax2_twin = axes[2].twinx()

            line1 = axes[2].plot(range(len(segments)), z_positions, 'b-o',
                                 alpha=0.7, markersize=3, label='Z-Position (mm)')
            line2 = ax2_twin.plot(range(len(segments)), feed_velocities, 'r-s',
                                  alpha=0.7, markersize=3, label='Vorschubgeschw. (mm/s)')

            axes[2].set_xlabel('Segment')
            axes[2].set_ylabel('Z-Position (mm)', color='b')
            ax2_twin.set_ylabel('Vorschubgeschwindigkeit (mm/s)', color='r')
            axes[2].set_title('Segment-Parameter')
            axes[2].grid(True, alpha=0.3)

            # Legende kombinieren
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[2].legend(lines, labels, loc='upper left')

        plt.tight_layout()
        plt.show()

        # Erweiterte Statistiken
        print(f"\n=== MRR Statistiken ===")
        print(f"Anzahl Segmente: {len(segments)}")
        print(f"Maximum MRR: {max_mrr:.3f} mm³/s")
        print(f"Mittlere MRR: {mean_mrr:.3f} mm³/s")
        print(f"Standardabweichung: {np.std(mrr_values):.3f} mm³/s")
        print(f"Gesamtvolumen entfernt: {np.trapz(mrr_values, times):.1f} mm³")
        print(f"Simulationszeit: {times[-1]:.2f} s")

        # Segment-Statistiken
        if segments:
            segment_lengths = [len(seg.positions) for seg in segments]
            print(f"\n=== Segment-Statistiken ===")
            print(f"Kleinster Segment: {min(segment_lengths)} Punkte")
            print(f"Größter Segment: {max(segment_lengths)} Punkte")
            print(f"Durchschnittlicher Segment: {np.mean(segment_lengths):.1f} Punkte")


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

    times, mrr_values, segments = simulator.simulate_mrr(sample_df, 50.0)

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