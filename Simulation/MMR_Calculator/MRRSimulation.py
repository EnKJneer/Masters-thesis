import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from typing import Tuple, Dict, Any


# JAX-kompilierte Hilfsfunktionen (außerhalb der Klasse)
@jax.jit
def _identify_support_points_jit(positions, velocities, spindle_speeds,
                                 eps_position, eps_velocity, part_position,
                                 part_width, part_depth, part_height, tool_radius):
    """JIT-kompilierte Version der Stützpunkt-Identifikation."""
    n_points = positions.shape[0]
    support_points = jnp.zeros(n_points, dtype=bool)

    # Anfangs- und Endpunkt
    support_points = support_points.at[0].set(True)
    support_points = support_points.at[-1].set(True)

    # Z-Position Änderungen
    z_diff = jnp.abs(jnp.diff(positions[:, 2]))
    z_changes = z_diff > eps_position
    support_points = support_points.at[1:].set(support_points[1:] | z_changes)

    # Geschwindigkeitsänderungen (xy-Ebene)
    velocity_magnitude = jnp.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
    velocity_diff = jnp.abs(jnp.diff(velocity_magnitude))
    velocity_changes = velocity_diff > eps_velocity
    support_points = support_points.at[1:].set(support_points[1:] | velocity_changes)

    # Spindeldrehzahl Änderungen (falls verfügbar)
    spindle_diff = jnp.abs(jnp.diff(spindle_speeds))
    spindle_changes = spindle_diff > (jnp.mean(spindle_speeds) * 0.05)  # 5% Änderung
    support_points = support_points.at[1:].set(support_points[1:] | spindle_changes)

    # Werkstück-Grenzen Check
    part_min = part_position
    part_max = part_position + jnp.array([part_width, part_depth, part_height])

    outside_bounds = ((positions < (part_min - tool_radius)) |
                      (positions > (part_max + tool_radius))).any(axis=1)
    boundary_transitions = jnp.diff(outside_bounds.astype(jnp.int32)) != 0
    support_points = support_points.at[1:].set(support_points[1:] | boundary_transitions)

    # Regelmäßige Abtastung (alle 100 Punkte)
    regular_sampling = jnp.arange(n_points) % 100 == 0
    support_points = support_points | regular_sampling

    return support_points


@jax.jit
def _world_to_voxel_jit(world_coords, part_position, voxel_size):
    """Konvertiert Weltkoordinaten zu Voxel-Indizes."""
    relative_coords = world_coords - part_position
    voxel_coords = relative_coords / voxel_size
    return jnp.round(voxel_coords).astype(jnp.int32)


@jax.jit
def _get_active_voxels_jit(tool_position, part_position, voxel_size, tool_radius, nx, ny, nz):
    """JIT-kompilierte Version der aktiven Voxel-Bestimmung."""
    tool_voxel = _world_to_voxel_jit(tool_position, part_position, voxel_size)

    # Define a fixed maximum safety radius
    max_safety_radius = 50

    # Create a grid for all possible relative indices
    range_vals = jnp.arange(-max_safety_radius, max_safety_radius + 1)
    dx, dy, dz = jnp.meshgrid(range_vals, range_vals, range_vals, indexing='ij')

    # Flatten the grid
    dx = dx.flatten()
    dy = dy.flatten()
    dz = dz.flatten()

    # Calculate absolute indices
    active_indices_x = tool_voxel[0] + dx
    active_indices_y = tool_voxel[1] + dy
    active_indices_z = tool_voxel[2] + dz

    # Calculate distances from tool position
    voxel_centers_x = active_indices_x * voxel_size + part_position[0] + voxel_size / 2
    voxel_centers_y = active_indices_y * voxel_size + part_position[1] + voxel_size / 2
    voxel_centers_z = active_indices_z * voxel_size + part_position[2] + voxel_size / 2

    distances = jnp.sqrt((voxel_centers_x - tool_position[0]) ** 2 +
                         (voxel_centers_y - tool_position[1]) ** 2 +
                         (voxel_centers_z - tool_position[2]) ** 2)

    # Safety distance: 2x tool radius
    safety_distance = 2 * tool_radius
    within_safety = distances <= safety_distance

    # Check if indices are within grid bounds
    valid_x = (active_indices_x >= 0) & (active_indices_x < nx)
    valid_y = (active_indices_y >= 0) & (active_indices_y < ny)
    valid_z = (active_indices_z >= 0) & (active_indices_z < nz)

    valid_mask = within_safety & valid_x & valid_y & valid_z

    # Use jnp.where to filter indices
    active_indices_x = jnp.where(valid_mask, active_indices_x, -1)
    active_indices_y = jnp.where(valid_mask, active_indices_y, -1)
    active_indices_z = jnp.where(valid_mask, active_indices_z, -1)

    # Stack the indices
    active_indices = jnp.stack([active_indices_x, active_indices_y, active_indices_z], axis=1)

    # Nur gültige Indizes herausfiltern, indem man `jnp.where` für die Maskierung nutzt
    # Stattdessen: alle Indices behalten, aber ungültige auf z. B. (-1, -1, -1) setzen
    invalid_voxel = jnp.array([-1, -1, -1], dtype=jnp.int32)
    active_indices = jnp.where(valid_mask[:, None], active_indices, invalid_voxel)

    return active_indices


@jax.jit
def _calculate_volume_removal_jit(tool_position, material_grid_state,
                                  part_position, voxel_size, tool_radius,
                                  nx, ny, nz):
    """JIT-kompilierte Version der Volumen-Entfernung."""
    active_voxels = _get_active_voxels_jit(tool_position, part_position, voxel_size,
                                           tool_radius, nx, ny, nz)

    # Leerer Fall
    def empty_case():
        return material_grid_state, 0.0

    # Fall mit aktiven Voxeln
    def active_case():
        # Voxel-Zentren in Weltkoordinaten
        voxel_centers = (active_voxels * voxel_size +
                         part_position + voxel_size / 2)

        # Werkzeug-Radius Check (Zylindrische Approximation in XY)
        tool_xy = tool_position[:2]
        voxel_xy = voxel_centers[:, :2]
        xy_distances = jnp.linalg.norm(voxel_xy - tool_xy, axis=1)

        # Z-Bereich Check (vereinfacht: Tool-Höhe = 3x Radius)
        tool_height = 3 * tool_radius
        z_in_range = (jnp.abs(voxel_centers[:, 2] - tool_position[2]) <= tool_height / 2)

        # Voxel im Werkzeug-Bereich
        inside_tool = (xy_distances <= tool_radius) & z_in_range

        # Aktuelles Material in diesen Voxeln
        current_material = material_grid_state[active_voxels[:, 0],
        active_voxels[:, 1],
        active_voxels[:, 2]]

        # Nur Material in Tool-Bereich entfernen
        material_to_remove = current_material * inside_tool

        # Material entfernen (setze betroffene Voxel auf 0)
        updated_grid = material_grid_state.at[active_voxels[:, 0],
        active_voxels[:, 1],
        active_voxels[:, 2]].set(
            current_material * (~inside_tool))

        # Entferntes Volumen berechnen
        removed_volume = jnp.sum(material_to_remove) * (voxel_size ** 3)

        return updated_grid, removed_volume

    return lax.cond(len(active_voxels) == 0, empty_case, active_case)


@jax.jit
def _scan_mrr_calculation(carry, inputs):
    """Scan-Funktion für MRR-Berechnung zwischen Stützpunkten."""
    material_grid, prev_pos, simulation_params = carry
    curr_pos, velocity, dt = inputs

    part_position, voxel_size, tool_radius, nx, ny, nz = simulation_params

    # Material vor der Bewegung
    prev_grid, _ = _calculate_volume_removal_jit(
        prev_pos, material_grid, part_position, voxel_size,
        tool_radius, nx, ny, nz
    )

    # Material nach der Bewegung
    updated_grid, volume_removed = _calculate_volume_removal_jit(
        curr_pos, prev_grid, part_position, voxel_size,
        tool_radius, nx, ny, nz
    )

    # MRR = Volumen / Zeit
    mrr = lax.cond(
        dt > 0,
        lambda: volume_removed / dt,
        lambda: 0.0
    )

    # Mit Vorschubgeschwindigkeit gewichten
    feed_velocity = jnp.linalg.norm(velocity[:2])  # XY-Geschwindigkeit
    mrr_weighted = mrr * feed_velocity

    # Update carry
    new_carry = (updated_grid, curr_pos, simulation_params)

    return new_carry, mrr_weighted


class CNCMRRSimulation:
    def __init__(self, part_position: list, part_dimension: list, tool_radius: float):
        """
        Initialisiert die CNC Material Removal Rate Simulation.

        Args:
            part_position: [x, y, z] Position des Werkstücks
            part_dimension: [width, depth, height, voxel_size]
            tool_radius: Werkzeugradius
        """
        self.part_position = jnp.array(part_position)
        self.part_width, self.part_depth, self.part_height, self.voxel_size = part_dimension
        self.tool_radius = tool_radius

        # Voxel-Grid Dimensionen berechnen
        self.nx = int(self.part_width / self.voxel_size)
        self.ny = int(self.part_depth / self.voxel_size)
        self.nz = int(self.part_height / self.voxel_size)

        # Schwellwerte für Stützpunktauswahl
        self.eps_velocity = 0.1
        self.eps_position = 1e-3

        # Werkstück-Grid initialisieren (1 = Material vorhanden, 0 = entfernt)
        self.material_grid = jnp.ones((self.nx, self.ny, self.nz), dtype=jnp.float32)

        print(f"Voxel-Grid initialisiert: {self.nx}×{self.ny}×{self.nz} = {self.nx * self.ny * self.nz:,} Voxel")

    def world_to_voxel(self, world_coords: jnp.ndarray) -> jnp.ndarray:
        """Konvertiert Weltkoordinaten zu Voxel-Indizes."""
        relative_coords = world_coords - self.part_position
        voxel_coords = relative_coords / self.voxel_size
        return jnp.round(voxel_coords).astype(jnp.int32)

    def identify_support_points(self, positions: jnp.ndarray, velocities: jnp.ndarray,
                                spindle_speeds: jnp.ndarray) -> jnp.ndarray:
        """
        Identifiziert Stützpunkte basierend auf den definierten Kriterien.

        Returns:
            Boolean array der Länge N mit True für Stützpunkte
        """
        return _identify_support_points_jit(
            positions, velocities, spindle_speeds,
            self.eps_position, self.eps_velocity,
            self.part_position, self.part_width, self.part_depth, self.part_height,
            self.tool_radius
        )

    def get_active_voxels(self, tool_position: jnp.ndarray) -> jnp.ndarray:
        """
        Bestimmt die aktiven Voxel um eine Werkzeugposition.

        Returns:
            Voxel-Indizes die vom Werkzeug beeinflusst werden könnten
        """
        return _get_active_voxels_jit(
            tool_position, self.part_position, self.voxel_size,
            self.tool_radius, self.nx, self.ny, self.nz
        )

    def apply_smoothing(self, mrr_values: np.ndarray, method: str = 'savgol', **kwargs) -> np.ndarray:
        """
        Wendet verschiedene Glättungsfilter auf die MRR-Werte an.

        Args:
            mrr_values: Rohe MRR-Werte
            method: Glättungsmethode ('savgol', 'gaussian', 'uniform', 'butterworth', 'rolling_median')
            **kwargs: Parameter für die jeweilige Methode

        Returns:
            Geglättete MRR-Werte
        """
        if len(mrr_values) < 10:
            print("Warnung: Zu wenige Datenpunkte für Glättung")
            return mrr_values

        if method == 'savgol':
            # Savitzky-Golay Filter (Standard)
            window_length = kwargs.get('window_length', min(51, len(mrr_values) // 4))
            if window_length % 2 == 0:
                window_length += 1  # Muss ungerade sein
            window_length = max(5, window_length)  # Mindestens 5
            polyorder = kwargs.get('polyorder', min(3, window_length - 2))

            return savgol_filter(mrr_values, window_length, polyorder)

        elif method == 'gaussian':
            # Gaussscher Glättungsfilter
            sigma = kwargs.get('sigma', len(mrr_values) * 0.01)  # 1% der Datenlänge
            return gaussian_filter1d(mrr_values, sigma)

        elif method == 'uniform':
            # Gleichmäßiger gleitender Durchschnitt
            window_size = kwargs.get('window_size', len(mrr_values) // 20)
            window_size = max(3, window_size)
            return uniform_filter1d(mrr_values, size=window_size)

        elif method == 'butterworth':
            # Butterworth Tiefpassfilter
            cutoff_freq = kwargs.get('cutoff_freq', 0.1)  # Normalisierte Frequenz (0-1)
            order = kwargs.get('order', 4)

            # Butterworth Filter Design
            b, a = butter(order, cutoff_freq, btype='low')
            return filtfilt(b, a, mrr_values)

        elif method == 'rolling_median':
            # Gleitender Median (robust gegen Ausreißer)
            window_size = kwargs.get('window_size', len(mrr_values) // 20)
            window_size = max(3, window_size)

            # Pandas rolling median
            df_temp = pd.DataFrame({'mrr': mrr_values})
            smoothed = df_temp['mrr'].rolling(window=window_size, center=True).median()
            # NaN-Werte am Rand mit ursprünglichen Werten füllen
            return smoothed.fillna(method='bfill').fillna(method='ffill').values

        elif method == 'adaptive':
            # Adaptive Glättung basierend auf lokaler Varianz
            window_size = kwargs.get('base_window_size', len(mrr_values) // 30)
            window_size = max(5, window_size)

            smoothed = np.copy(mrr_values)

            for i in range(len(mrr_values)):
                # Lokales Fenster definieren
                start = max(0, i - window_size // 2)
                end = min(len(mrr_values), i + window_size // 2 + 1)

                local_data = mrr_values[start:end]
                local_std = np.std(local_data)

                # Adaptive Gewichtung basierend auf Standardabweichung
                if local_std > np.std(mrr_values) * 0.5:  # Hochvariable Bereiche
                    # Starke Glättung
                    weight = 0.3
                else:  # Stabile Bereiche
                    # Leichte Glättung
                    weight = 0.8

                smoothed[i] = weight * mrr_values[i] + (1 - weight) * np.mean(local_data)

            return smoothed

        else:
            raise ValueError(f"Unbekannte Glättungsmethode: {method}")

    def calculate_volume_removal(self, tool_position: jnp.ndarray,
                                 material_grid_state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.float32]:
        """
        Berechnet das entfernte Volumen für eine Werkzeugposition.

        Returns:
            (updated_material_grid, removed_volume)
        """
        return _calculate_volume_removal_jit(
            tool_position, material_grid_state,
            self.part_position, self.voxel_size, self.tool_radius,
            self.nx, self.ny, self.nz
        )

    def simulate_mrr(self, df: pd.DataFrame, smooth_method: str = 'rolling_median',
                     smooth_params: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hauptsimulation der Material Removal Rate mit jax.scan und Glättung.

        Args:
            df: DataFrame mit Spalten ['pos_x', 'pos_y', 'pos_z', 'pos_sp', 'v_x', 'v_y', 'v_z', 'v_sp']
            smooth_method: Glättungsmethode ('savgol', 'gaussian', 'uniform', 'butterworth', 'rolling_median', 'adaptive', 'none')
            smooth_params: Parameter für die Glättungsmethode

        Returns:
            (times, mrr_values_smoothed)
        """
        print("Konvertiere DataFrame zu JAX Arrays...")

        # Daten extrahieren
        positions = jnp.array(df[['pos_x', 'pos_y', 'pos_z']].values, dtype=jnp.float32)
        velocities = jnp.array(df[['v_x', 'v_y', 'v_z']].values, dtype=jnp.float32)
        spindle_speeds = jnp.array(df['pos_sp'].values, dtype=jnp.float32) if 'pos_sp' in df else None

        n_points = len(positions)
        times = np.arange(n_points, dtype=np.float32)  # Konstante Zeitschritte

        print("Identifiziere Stützpunkte...")
        support_mask = self.identify_support_points(positions, velocities, spindle_speeds)
        support_indices = np.where(support_mask)[0]

        print(f"Gefunden: {len(support_indices)} Stützpunkte von {n_points} Gesamtpunkten")

        if len(support_indices) < 2:
            print("Warnung: Zu wenige Stützpunkte gefunden. Verwende alle Punkte.")
            support_indices = np.arange(n_points)

        # Daten für jax.scan vorbereiten
        print("Bereite Daten für JAX scan vor...")

        # Aktuelle und vorherige Positionen
        curr_positions = positions[support_indices[1:]]  # Start ab zweitem Punkt
        prev_positions = positions[support_indices[:-1]]  # Alle außer letztem
        curr_velocities = velocities[support_indices[1:]]

        # Zeitdifferenzen zwischen Stützpunkten
        time_diffs = jnp.diff(support_indices.astype(jnp.float32))

        # Simulation parameter tupel
        simulation_params = (
            self.part_position, self.voxel_size, self.tool_radius,
            self.nx, self.ny, self.nz
        )

        # Initialer Zustand (carry)
        initial_carry = (
            self.material_grid,  # Material grid
            positions[support_indices[0]],  # Erste Position
            simulation_params
        )

        # Eingabedaten für scan (xs)
        scan_inputs = (curr_positions, curr_velocities, time_diffs)

        print("Führe JAX scan für MRR-Berechnung aus...")
        print("Kompiliere JAX-Funktionen (kann beim ersten Mal dauern)...")

        # Progressbar-Simulation durch Chunking
        chunk_size = max(1, len(support_indices) // 20)  # 20 Updates
        total_chunks = (len(support_indices) - 1 + chunk_size - 1) // chunk_size

        all_mrr_values = []
        current_carry = initial_carry

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(support_indices) - 1)

            if start_idx >= end_idx:
                break

            # Chunk-Daten extrahieren
            chunk_positions = curr_positions[start_idx:end_idx]
            chunk_velocities = curr_velocities[start_idx:end_idx]
            chunk_time_diffs = time_diffs[start_idx:end_idx]

            chunk_inputs = (chunk_positions, chunk_velocities, chunk_time_diffs)

            # Scan für diesen Chunk
            final_carry, chunk_mrr = jax.lax.scan(_scan_mrr_calculation, current_carry, chunk_inputs)

            # Ergebnisse sammeln
            all_mrr_values.append(chunk_mrr)
            current_carry = final_carry

            # Fortschritt anzeigen
            progress = (chunk_idx + 1) / total_chunks * 100
            print(f"  Fortschritt: {progress:.1f}% ({chunk_idx + 1}/{total_chunks} Chunks)")

        # Alle MRR-Werte zusammenführen
        if all_mrr_values:
            support_mrr = jnp.concatenate(all_mrr_values)
            # Ersten Punkt (MRR=0) hinzufügen
            support_mrr = jnp.concatenate([jnp.array([0.0]), support_mrr])
        else:
            support_mrr = jnp.zeros(len(support_indices))

        # Konvertiere zurück zu numpy für Interpolation
        support_mrr = np.array(support_mrr)

        print("Interpoliere MRR-Werte...")
        '''
        # Spline-Interpolation zwischen Stützpunkten
        if len(support_indices) > 3:
            # Cubic Spline
            cs = CubicSpline(support_indices, support_mrr, bc_type='natural')
            mrr_interpolated = cs(times)
        else:'''
        # Linear interpolation
        mrr_interpolated = np.interp(times, support_indices, support_mrr)

        # Negative Werte auf 0 setzen
        mrr_interpolated = np.maximum(mrr_interpolated, 0.0)

        # Glättung anwenden
        if smooth_method != 'none':
            print(f"Wende {smooth_method} Glättung an...")
            if smooth_params is None:
                smooth_params = {}
            mrr_smoothed = self.apply_smoothing(mrr_interpolated, smooth_method, **smooth_params)
        else:
            mrr_smoothed = mrr_interpolated

        print("Simulation abgeschlossen!")
        return times, mrr_smoothed

    def plot_results(self, times: np.ndarray, mrr_values: np.ndarray,
                     df: pd.DataFrame = None, show_raw: bool = False, raw_mrr: np.ndarray = None):
        """
        Plottet die MRR-Ergebnisse.

        Args:
            times: Zeitarray
            mrr_values: Geglättete MRR-Werte
            df: Original DataFrame (optional)
            show_raw: Ob rohe Daten auch gezeigt werden sollen
            raw_mrr: Rohe MRR-Werte (optional)
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # MRR über Zeit
        if show_raw and raw_mrr is not None:
            axes[0].plot(times, raw_mrr, 'lightgray', linewidth=0.5, alpha=0.7, label='Roh')

        axes[0].plot(times, mrr_values, 'b-', linewidth=1.5, alpha=0.9, label='Geglättet')
        axes[0].set_xlabel('Zeit (Zeitschritte)')
        axes[0].set_ylabel('MRR (mm³/Zeitschritt)')
        axes[0].set_title('Material Removal Rate über Zeit')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Statistiken anzeigen
        max_mrr = np.max(mrr_values)
        mean_mrr = np.mean(mrr_values)
        axes[0].axhline(mean_mrr, color='r', linestyle='--', alpha=0.7,
                        label=f'Mittelwert: {mean_mrr:.3f}')
        axes[0].legend()

        # Geschwindigkeit über Zeit (falls DataFrame verfügbar)
        if df is not None:
            velocity_magnitude = np.sqrt(df['v_x'] ** 2 + df['v_y'] ** 2)
            axes[1].plot(times[:len(velocity_magnitude)], velocity_magnitude, 'g-',
                         linewidth=1, alpha=0.8)
            axes[1].set_xlabel('Zeit (Zeitschritte)')
            axes[1].set_ylabel('Vorschubgeschwindigkeit (mm/Zeitschritt)')
            axes[1].set_title('Vorschubgeschwindigkeit über Zeit')
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Statistiken ausgeben
        print(f"\n=== MRR Statistiken (Geglättet) ===")
        print(f"Maximum MRR: {max_mrr:.3f} mm³/Zeitschritt")
        print(f"Mittlere MRR: {mean_mrr:.3f} mm³/Zeitschritt")
        print(f"Standardabweichung: {np.std(mrr_values):.3f} mm³/Zeitschritt")
        print(f"Gesamtvolumen entfernt: {np.sum(mrr_values):.1f} mm³")

        if show_raw and raw_mrr is not None:
            print(f"\n=== Vergleich Roh vs. Geglättet ===")
            print(f"Rauschreduktion (Std): {np.std(raw_mrr):.3f} → {np.std(mrr_values):.3f} "
                  f"({(1 - np.std(mrr_values) / np.std(raw_mrr)) * 100:.1f}% Reduktion)")
            print(f"Signal-Rausch-Verhältnis: {mean_mrr / np.std(mrr_values):.2f}")

    def analyze_and_plot_smoothing(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Führt MRR-Simulation durch und vergleicht verschiedene Glättungsmethoden.

        Returns:
            (times, raw_mrr, smoothed_methods_dict)
        """
        # Erst ohne Glättung simulieren
        times, raw_mrr = self.simulate_mrr(df, smooth_method='none')

        # Verschiedene Glättungsmethoden vergleichen
        print("\nVergleiche Glättungsmethoden...")
        smoothed_methods = self.compare_smoothing_methods(raw_mrr, times)

        return times, raw_mrr, smoothed_methods

    def compare_smoothing_methods(self, raw_mrr: np.ndarray, times: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vergleicht verschiedene Glättungsmethoden auf die Roh-MRR-Daten.

        Args:
            raw_mrr: Roh-MRR-Werte
            times: Zeitstempel für die MRR-Werte

        Returns:
            Ein Dictionary mit den geglätteten MRR-Werten für jede Methode.
        """
        smoothing_methods = {
            'savgol': {'window_length': min(51, len(raw_mrr) // 4), 'polyorder': 3},
            'gaussian': {'sigma': len(raw_mrr) * 0.01},
            'uniform': {'window_size': len(raw_mrr) // 20},
            'butterworth': {'cutoff_freq': 0.1, 'order': 4},
            'rolling_median': {'window_size': len(raw_mrr) // 100},
            'adaptive': {'base_window_size': len(raw_mrr) // 30}
        }

        smoothed_results = {}

        for method, params in smoothing_methods.items():
            print(f"Anwendung der {method} Glättungsmethode...")
            smoothed_mrr = self.apply_smoothing(raw_mrr, method, **params)
            smoothed_results[method] = smoothed_mrr

        return smoothed_results

def choose_best_smoothing_method(smoothed_methods, raw_mrr):
    best_method = None
    best_std_reduction = float('inf')

    for method, smoothed_mrr in smoothed_methods.items():
        # Berechne die Standardabweichung der geglätteten Daten
        std_smoothed = np.std(smoothed_mrr)
        std_raw = np.std(raw_mrr)

        # Berechne die Reduktion der Standardabweichung
        std_reduction = std_smoothed / std_raw

        # Wähle die Methode mit der geringsten Standardabweichungsreduktion
        if std_reduction < best_std_reduction:
            best_std_reduction = std_reduction
            best_method = method

    return best_method

# Beispiel-Nutzung
if __name__ == "__main__":
    # Beispiel-Parameter
    part_position = [-35.64, 175.0, 354.94]
    part_dimension = [75.0, 75.0 * 2, 50.0, 0.1]  # width, depth, height, voxel_size
    tool_radius = 5.0  # mm

    # Beispiel-DataFrame erstellen (in der Praxis aus echten Daten laden)
    n_points = 1000
    t = np.linspace(0, 10, n_points)

    # Simulierte Toolpath (Spirale)
    path_data = '..\\..\\DataSetsV3/DataMerged/S235JR_Plate_Normal.csv'
    sample_df = pd.read_csv(path_data)
    n = int(len(sample_df) / 3)
    sample_df = sample_df[:n]
    # Simulation ausführen
    print("Starte CNC MRR Simulation...")
    simulator = CNCMRRSimulation(part_position, part_dimension, tool_radius)

    # Option 1: Direkt mit Glättung simulieren
    times, mrr_values = simulator.simulate_mrr(sample_df, smooth_method='savgol')
    simulator.plot_results(times, mrr_values, sample_df)

    # Option 2: Verschiedene Glättungsmethoden vergleichen
    print("\n" + "=" * 50)
    print("Vergleiche verschiedene Glättungsmethoden:")
    times_raw, raw_mrr, smoothed_methods = simulator.analyze_and_plot_smoothing(sample_df)

    # Beste Methode auswählen und nochmal plotten
    best_method = choose_best_smoothing_method(smoothed_methods, raw_mrr)
    print(f'Beste Methode: {best_method}')
    simulator.plot_results(times_raw, smoothed_methods[best_method], sample_df,
                           show_raw=True, raw_mrr=raw_mrr)