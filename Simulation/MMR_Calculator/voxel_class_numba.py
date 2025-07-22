import numpy as np
from numba import njit, prange
from typing import List, Optional
import jax.numpy as jnp
from jax import jit, vmap, lax
from typing import NamedTuple

class Tool:
    """
    Klasse Tool. Definiert ein Werkzeug mit Durchmesser und Koordinaten des Tool-Center-Point.
    """

    def __init__(self, radius: float, coordinates: List[float]) -> None:
        if len(coordinates) != 3:
            raise ValueError("Coordinates must be a list of three elements.")
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self.radius = radius
        self.coordinates = coordinates
        self.x_position, self.y_position, self.z_position = coordinates

    def set_new_position(self, coordinates: List[float]) -> None:
        if len(coordinates) != 3:
            raise ValueError("Coordinates must be a list of three elements.")
        self.x_position, self.y_position, self.z_position = coordinates

    def get_ap(self, old_ap: float) -> float:
        a_p = old_ap - self.z_position
        return a_p

class Part:
    """
    Part-Klasse repraesentiert ein 3D-Bauteil bestehend aus Voxeln.
    """

    def __init__(self, width: float, depth: float, height: float, voxel_size: float, coordinates: Optional[List[float]] = None) -> None:
        if coordinates is None:
            coordinates = [0.0, 0.0, 0.0]

        self.width = width
        self.height = height
        self.depth = depth
        self.voxel_size = voxel_size
        self.coordinates = coordinates

        self.x_origin, self.y_origin, self.z_origin = coordinates

        # Berechnung der Anzahl der Voxel in jeder Dimension
        self.x_voxels = int(round(width / voxel_size))
        self.y_voxels = int(round(depth / voxel_size))
        self.z_voxels = int(round(height / voxel_size))

        # Initialisierung des Fill-Status als boolesches Numpy-Array
        self.voxels_fill = np.ones((self.x_voxels, self.y_voxels, self.z_voxels), dtype=np.bool_)
        self.history = None

    def insert_circle_centered(self, radius: int):
        """
        Fuegt einen kreisfoermigen Querschnitt in der XY-Ebene fuer alle Z-Schichten ein.
        Der Kreis ist immer im Mittelpunkt der XY-Ebene positioniert.
    
        Parameters:
        - self: 3D NumPy-Array
        - radius: Radius des Kreises
    
        Die Voxel innerhalb des Radius werden auf True gesetzt, der Rand bleibt False.
        """

        x_dim, y_dim, z_dim = self.voxels_fill.shape
        # Berechne den Mittelpunkt basierend auf den Dimensionen
        x0 = x_dim // 2
        y0 = y_dim // 2
    
        # Erstelle ein 2D-Gitter fuer die XY-Ebene
        Y, X = np.ogrid[:y_dim, :x_dim]
        distance = np.sqrt((X - x0)**2 + (Y - y0)**2)
    
        # Erstelle eine Maske fuer den Kreis (innerhalb des Radius)
        mask = distance <= radius
    
        # Setze die Voxel innerhalb des Kreises auf True fuer alle Z-Schichten
        # Dies kann effizient ohne Schleifen erfolgen
        self.voxels_fill[:, :, :] = False  # Setze zunaechst alle Voxel auf False
        self.voxels_fill[mask, :] = True     # Setze alle Voxel innerhalb des Kreises auf True fuer alle Z

    def calculate_smoothed_material(self, raw_value, history, alpha=0.3):
        if not history:
            return raw_value
        return alpha * raw_value + (1 - alpha) * history[-1]

    def get_total_volume(self) -> float:
        """
        Berechnet das gesamte Volumen der gefuellten Voxel.
        """
        new = np.sum(self.voxels_fill) * (self.voxel_size ** 3)

        return self.calculate_smoothed_material(new, self.history)

    def apply_tool(self, tool: Tool) -> None:
        """
        Setzt das Werkzeug auf das Bauteil. Alle Voxel-Elemente, welche das Werkzeug berühren,
        werden als entfernt identifiziert durch das Setzen von fill auf False.
        :param tool: Ein definiertes Werkzeug aus der Tool-Klasse.
        """
        # Berechne die Position der Tool in Z-Achse
        tool_z_penetration = tool.z_position / self.voxel_size
        # Annahme: Das Werkzeug steht senkrecht zum Werkstück und ist nicht verdreht in x- oder y-Achse
        relative_penetration = tool_z_penetration - (self.z_origin / self.voxel_size + self.z_voxels)

        if relative_penetration >= 0:
            return

        affected_z_index = int(round(tool_z_penetration) - self.z_origin / self.voxel_size)
        end_z_index = self.z_voxels

        # Vektorisierte Anwendung des Werkzeugs auf alle betroffenen Z-Schichten
        self.voxels_fill = self.apply_tool_vectorized(
            self.voxels_fill,
            self.x_origin,
            self.y_origin,
            self.voxel_size,
            tool.x_position,
            tool.y_position,
            tool.radius,
            affected_z_index,
            end_z_index
        )

    @staticmethod
    @jit
    def apply_tool_vectorized(fill_layer, x_origin, y_origin, voxel_size, tool_x, tool_y, radius, affected_z_index,
                                 end_z_index):
        """
        Vektorisierte Funktion zur Anwendung des Werkzeugs auf die Voxel.
        Uses direct boolean indexing - most efficient approach.
        """
        x_voxels, y_voxels, z_voxels = fill_layer.shape

        # Erstelle Gitter für die Voxel-Koordinaten
        i_indices = jnp.arange(x_voxels)[:, None, None]
        j_indices = jnp.arange(y_voxels)[None, :, None]
        k_indices = jnp.arange(z_voxels)[None, None, :]

        voxel_x = x_origin + i_indices * voxel_size
        voxel_y = y_origin + j_indices * voxel_size

        # Finde die nächsten Punkte auf den Voxeln zum Werkzeugzentrum
        nearest_x = jnp.maximum(voxel_x, jnp.minimum(tool_x, voxel_x + voxel_size))
        nearest_y = jnp.maximum(voxel_y, jnp.minimum(tool_y, voxel_y + voxel_size))

        # Berechne die Quadrat-Distanz
        dist_x = tool_x - nearest_x
        dist_y = tool_y - nearest_y
        dist_sq = dist_x ** 2 + dist_y ** 2

        # Setze die Voxel auf False, wenn sie innerhalb des Radius liegen und innerhalb der betroffenen Z-Indizes
        mask = (dist_sq < radius ** 2) & (k_indices >= affected_z_index) & (k_indices < end_z_index)

        # Direct boolean indexing - JAX handles this efficiently
        fill_layer = jnp.where(mask, False, fill_layer)

        return fill_layer


class PartialVoxelCalculator:
    """
    Klasse für die Berechnung partieller Voxel-Überlappungen mit Werkzeugen
    """

    @staticmethod
    def sphere_box_intersection_volume(sphere_center, sphere_radius, box_min, box_max):
        """
        Berechnet das Überlappungsvolumen zwischen einer Kugel (Werkzeug) und einem Box (Voxel).

        Args:
            sphere_center: (x, y, z) Zentrum der Kugel
            sphere_radius: Radius der Kugel
            box_min: (x_min, y_min, z_min) untere Ecke der Box
            box_max: (x_max, y_max, z_max) obere Ecke der Box

        Returns:
            float: Überlappungsvolumen (0.0 bis voxel_volume)
        """
        # Vereinfachte analytische Berechnung für Kugel-Box Überlappung
        # Für präzise Berechnung wäre Monte-Carlo oder numerische Integration nötig

        # Prüfe ob Kugel die Box überhaupt berührt
        closest_point = jnp.array([
            jnp.clip(sphere_center[0], box_min[0], box_max[0]),
            jnp.clip(sphere_center[1], box_min[1], box_max[1]),
            jnp.clip(sphere_center[2], box_min[2], box_max[2])
        ])

        distance = jnp.linalg.norm(sphere_center - closest_point)

        # Keine Überlappung
        if distance > sphere_radius:
            return 0.0

        # Vollständige Überlappung (Voxel komplett innerhalb der Kugel)
        box_size = box_max - box_min
        box_volume = jnp.prod(box_size)

        # Prüfe ob alle Box-Ecken innerhalb der Kugel liegen
        corners = jnp.array([
            [box_min[0], box_min[1], box_min[2]],
            [box_min[0], box_min[1], box_max[2]],
            [box_min[0], box_max[1], box_min[2]],
            [box_min[0], box_max[1], box_max[2]],
            [box_max[0], box_min[1], box_min[2]],
            [box_max[0], box_min[1], box_max[2]],
            [box_max[0], box_max[1], box_min[2]],
            [box_max[0], box_max[1], box_max[2]]
        ])

        distances_to_corners = jnp.linalg.norm(corners - sphere_center[None, :], axis=1)
        all_corners_inside = jnp.all(distances_to_corners <= sphere_radius)

        if all_corners_inside:
            return box_volume

        # Partielle Überlappung - verwende Approximation
        return PartialVoxelCalculator._approximate_partial_overlap(
            sphere_center, sphere_radius, box_min, box_max, box_volume
        )

    @staticmethod
    def _approximate_partial_overlap(sphere_center, sphere_radius, box_min, box_max, box_volume):
        """
        Approximiert partielle Überlappung basierend auf dem Abstand zum Voxel-Zentrum
        """
        box_center = (box_min + box_max) / 2
        box_diagonal = jnp.linalg.norm(box_max - box_min) / 2

        distance_to_center = jnp.linalg.norm(sphere_center - box_center)

        # Lineare Interpolation basierend auf Distanz
        if distance_to_center <= sphere_radius - box_diagonal:
            # Zentrum weit innerhalb der Kugel
            overlap_ratio = 1.0
        elif distance_to_center >= sphere_radius + box_diagonal:
            # Zentrum weit außerhalb der Kugel
            overlap_ratio = 0.0
        else:
            # Partielle Überlappung - lineare Interpolation
            overlap_ratio = (sphere_radius + box_diagonal - distance_to_center) / (2 * box_diagonal)
            overlap_ratio = jnp.clip(overlap_ratio, 0.0, 1.0)

        return overlap_ratio * box_volume

class PartialVoxelPart:
    """
    Erweiterte Voxel-Klasse mit partiellen Füllständen statt binären Werten
    """

    def __init__(self, width: float, depth: float, height: float, voxel_size: float,
                 coordinates: Optional[List[float]] = None) -> None:
        if coordinates is None:
            coordinates = [0.0, 0.0, 0.0]

        self.width = width
        self.height = height
        self.depth = depth
        self.voxel_size = voxel_size
        self.coordinates = coordinates

        self.x_origin, self.y_origin, self.z_origin = coordinates

        # Berechnung der Anzahl der Voxel in jeder Dimension
        self.x_voxels = int(round(width / voxel_size))
        self.y_voxels = int(round(depth / voxel_size))
        self.z_voxels = int(round(height / voxel_size))

        # Jetzt float-Array statt boolean (0.0 = leer, 1.0 = voll)
        self.voxels_fill = jnp.ones((self.x_voxels, self.y_voxels, self.z_voxels), dtype=jnp.float32)

    def apply_tool(self, tool) -> None:
        """
        Wendet Werkzeug mit partieller Überlappungsberechnung an
        """
        # Berechne die Position der Tool in Z-Achse
        tool_z_penetration = tool.z_position / self.voxel_size
        relative_penetration = tool_z_penetration - (self.z_origin / self.voxel_size + self.z_voxels)

        if relative_penetration >= 0:
            return

        affected_z_index = int(round(tool_z_penetration) - self.z_origin / self.voxel_size)
        end_z_index = self.z_voxels

        # Anwenden der partiellen Überlappungsberechnung
        self.voxels_fill = self.apply_tool_partial_vectorized(
            self.voxels_fill,
            self.x_origin,
            self.y_origin,
            self.z_origin,
            self.voxel_size,
            tool.x_position,
            tool.y_position,
            tool.z_position,
            tool.radius,
            affected_z_index,
            end_z_index
        )

    @staticmethod
    @jit
    def apply_tool_partial_vectorized(fill_layer, x_origin, y_origin, z_origin, voxel_size,
                                      tool_x, tool_y, tool_z, radius, affected_z_index, end_z_index):
        """
        Vektorisierte partielle Überlappungsberechnung
        """
        x_voxels, y_voxels, z_voxels = fill_layer.shape

        # Erstelle Gitter für die Voxel-Koordinaten
        i_indices = jnp.arange(x_voxels)[:, None, None]
        j_indices = jnp.arange(y_voxels)[None, :, None]
        k_indices = jnp.arange(z_voxels)[None, None, :]

        # Berechne Voxel-Positionen (Zentren)
        voxel_x = x_origin + (i_indices + 0.5) * voxel_size
        voxel_y = y_origin + (j_indices + 0.5) * voxel_size
        voxel_z = z_origin + (k_indices + 0.5) * voxel_size

        # Nur Voxel in relevanten Z-Bereichen betrachten
        z_mask = (k_indices >= affected_z_index) & (k_indices < end_z_index)

        # Berechne Distanz von Voxel-Zentrum zum Werkzeug-Zentrum
        dist_to_tool = jnp.sqrt(
            (voxel_x - tool_x) ** 2 +
            (voxel_y - tool_y) ** 2 +
            (voxel_z - tool_z) ** 2
        )

        # Definiere Voxel-Halbdiagonale für Überlappungsberechnung
        voxel_half_diagonal = voxel_size * jnp.sqrt(3) / 2

        # Berechne Überlappungsfaktor
        overlap_factor = jnp.where(
            z_mask,
            jnp.where(
                dist_to_tool <= radius - voxel_half_diagonal,
                1.0,  # Vollständig innerhalb
                jnp.where(
                    dist_to_tool >= radius + voxel_half_diagonal,
                    0.0,  # Vollständig außerhalb
                    # Partielle Überlappung (lineare Approximation)
                    (radius + voxel_half_diagonal - dist_to_tool) / (2 * voxel_half_diagonal)
                )
            ),
            0.0  # Nicht in Z-Bereich
        )

        # Clamp auf [0, 1]
        overlap_factor = jnp.clip(overlap_factor, 0.0, 1.0)

        # Reduziere Voxel-Füllung basierend auf Überlappung
        new_fill = fill_layer * (1.0 - overlap_factor)

        return new_fill

    def calculate_smoothed_material(self, raw_value, history, alpha=0.3):
        if not history:
            return raw_value
        return alpha * raw_value + (1 - alpha) * history[-1]

    def get_total_volume(self) -> float:
        """
        Berechnet das gesamte Volumen mit partiellen Füllständen
        """
        return float(jnp.sum(self.voxels_fill) * (self.voxel_size ** 3))

    def get_material_density_stats(self):
        """
        Statistiken über die Materialverteilung
        """
        return {
            'min_fill': float(jnp.min(self.voxels_fill)),
            'max_fill': float(jnp.max(self.voxels_fill)),
            'mean_fill': float(jnp.mean(self.voxels_fill)),
            'std_fill': float(jnp.std(self.voxels_fill)),
            'total_volume': self.get_total_volume(),
            'empty_voxels': int(jnp.sum(self.voxels_fill == 0.0)),
            'full_voxels': int(jnp.sum(self.voxels_fill == 1.0)),
            'partial_voxels': int(jnp.sum((self.voxels_fill > 0.0) & (self.voxels_fill < 1.0)))
        }


