import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
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

    # Regelmäßige Abtastung (alle 50 Punkte)
    regular_sampling = jnp.arange(n_points) % 50 == 0
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

    distances = jnp.sqrt((voxel_centers_x - tool_position[0])**2 +
                         (voxel_centers_y - tool_position[1])**2 +
                         (voxel_centers_z - tool_position[2])**2)

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
    # Stattdessen: alle Indices behalten, aber ungültige auf z. B. (-1, -1, -1) setzen
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

    def simulate_mrr(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Hauptsimulation der Material Removal Rate.

        Args:
            df: DataFrame mit Spalten ['pos_x', 'pos_y', 'pos_z', 'pos_sp', 'v_x', 'v_y', 'v_z', 'v_sp']

        Returns:
            (times, mrr_values)
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

        # MRR für Stützpunkte berechnen
        print("Berechne MRR für Stützpunkte...")
        support_mrr = []
        current_material_grid = self.material_grid

        for i, idx in enumerate(support_indices):
            if i == 0:
                # Erster Punkt: keine Änderung
                mrr = 0.0
            else:
                # Vorherige und aktuelle Position
                prev_pos = positions[support_indices[i - 1]]
                curr_pos = positions[idx]

                # Material vor der Bewegung
                prev_grid, _ = self.calculate_volume_removal(prev_pos, current_material_grid)

                # Material nach der Bewegung
                updated_grid, volume_removed = self.calculate_volume_removal(curr_pos, prev_grid)
                current_material_grid = updated_grid

                # Zeit zwischen Stützpunkten
                dt = support_indices[i] - support_indices[i - 1]

                # MRR = Volumen / Zeit
                mrr = volume_removed / max(dt, 1) if dt > 0 else 0.0

                # Mit Vorschubgeschwindigkeit gewichten
                feed_velocity = jnp.linalg.norm(velocities[idx, :2])  # XY-Geschwindigkeit
                mrr = mrr * feed_velocity

            support_mrr.append(float(mrr))

            if (i + 1) % 100 == 0:
                print(f"  Bearbeitet: {i + 1}/{len(support_indices)} Stützpunkte")

        support_mrr = np.array(support_mrr)

        print("Interpoliere MRR-Werte...")
        # Spline-Interpolation zwischen Stützpunkten
        if len(support_indices) > 3:
            # Cubic Spline
            cs = CubicSpline(support_indices, support_mrr, bc_type='natural')
            mrr_interpolated = cs(times)
        else:
            # Linear interpolation fallback
            mrr_interpolated = np.interp(times, support_indices, support_mrr)

        # Negative Werte auf 0 setzen
        mrr_interpolated = np.maximum(mrr_interpolated, 0.0)

        print("Simulation abgeschlossen!")
        return times, mrr_interpolated

    def plot_results(self, times: np.ndarray, mrr_values: np.ndarray,
                     df: pd.DataFrame = None):
        """Plottet die MRR-Ergebnisse."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # MRR über Zeit
        axes[0].plot(times, mrr_values, 'b-', linewidth=1, alpha=0.8)
        axes[0].set_xlabel('Zeit (Zeitschritte)')
        axes[0].set_ylabel('MRR (mm³/Zeitschritt)')
        axes[0].set_title('Material Removal Rate über Zeit')
        axes[0].grid(True, alpha=0.3)

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
        print(f"\n=== MRR Statistiken ===")
        print(f"Maximum MRR: {max_mrr:.3f} mm³/Zeitschritt")
        print(f"Mittlere MRR: {mean_mrr:.3f} mm³/Zeitschritt")
        print(f"Gesamtvolumen entfernt: {np.sum(mrr_values):.1f} mm³")


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
    sample_df = pd.DataFrame({
        'pos_x': part_position[0] + 20 * np.cos(t) * (1 - t / 10),
        'pos_y': part_position[1] + 20 * np.sin(t) * (1 - t / 10),
        'pos_z': part_position[2] + 10 - t,  # Z nach unten
        'pos_sp': 12000 + 1000 * np.sin(t * 2),  # Spindeldrehzahl mit Rauschen
        'v_x': np.gradient(20 * np.cos(t) * (1 - t / 10)),
        'v_y': np.gradient(20 * np.sin(t) * (1 - t / 10)),
        'v_z': -np.ones(n_points),  # Konstante Z-Geschwindigkeit
        'v_sp': np.gradient(12000 + 1000 * np.sin(t * 2))
    })

    # Simulation ausführen
    print("Starte CNC MRR Simulation...")
    simulator = CNCMRRSimulation(part_position, part_dimension, tool_radius)
    times, mrr_values = simulator.simulate_mrr(sample_df)

    # Ergebnisse plotten
    simulator.plot_results(times, mrr_values, sample_df)