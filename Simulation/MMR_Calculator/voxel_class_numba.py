import numpy as np
from numba import njit, prange
from typing import List, Optional

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

    def get_total_volume(self) -> float:
        """
        Berechnet das gesamte Volumen der gefuellten Voxel.
        """
        return np.sum(self.voxels_fill) * (self.voxel_size ** 3)

    def apply_tool(self, tool: Tool) -> None:
        """
        Setzt das Werkzeug auf das Bauteil. Alle Voxel-Elemente, welche das Werkzeug beruehren,
        werden als entfernt identifiziert durch das Setzen von fill auf False.
        :param tool: Ein definiertes Werkzeug aus der Tool-Klasse.
        """
        # Berechne die Position der Tool in Z-Achse
        tool_z_penetration = tool.z_position/self.voxel_size

        # Annahme: Das Werkzeug steht senkrecht zum Werkstueck und ist nicht verdreht in x- oder y-Achse
        relative_penetration = tool_z_penetration - (self.z_origin / self.voxel_size + self.z_voxels)#Nur wenn das hier < 0 ist dann befindet sich Tool in Part

        affected_z_index = int(round(tool_z_penetration) - self.z_origin / self.voxel_size)

        # Sicherstellen, dass wir nicht ueber die Grenzen hinausgehen
        end_z_index = self.z_voxels
        if relative_penetration >= 0:
            #print(f"Werkzeugposition {tool_z_penetration} liegt ausserhalb des Bauteils.")
            pass
        else:
            for z in range(affected_z_index, end_z_index):
                apply_tool_numba(
                    self.voxels_fill[:, :, z],
                    self.x_origin,
                    self.y_origin,
                    self.voxel_size,
                    tool.x_position,
                    tool.y_position,
                    tool.radius
                )

@njit(parallel=True)
def apply_tool_numba(fill_layer, x_origin, y_origin, voxel_size, tool_x, tool_y, tool_radius):
    """
    Numba-optimierte Funktion zur Anwendung des Werkzeugs auf eine Schicht von Voxel-Fill-Status.
    Setzt den Fill-Status auf False fuer Voxel, die innerhalb des Werkzeugradius liegen.
    """
    x_voxels, y_voxels = fill_layer.shape
    radius_sq = tool_radius ** 2

    for i in prange(x_voxels):
        for j in range(y_voxels):
            if fill_layer[i, j]:
                # Berechne die Koordinaten des aktuellen Voxels
                voxel_x = x_origin + i * voxel_size
                voxel_y = y_origin + j * voxel_size

                # Finde den nächsten Punkt auf dem Voxel zum Werkzeugzentrum
                nearest_x = max(voxel_x, min(tool_x, voxel_x + voxel_size))
                nearest_y = max(voxel_y, min(tool_y, voxel_y + voxel_size))

                # Berechne die Quadrat-Distanz
                dist_x = tool_x - nearest_x
                dist_y = tool_y - nearest_y
                dist_sq = dist_x ** 2 + dist_y ** 2

                if dist_sq < radius_sq:
                    fill_layer[i, j] = False
