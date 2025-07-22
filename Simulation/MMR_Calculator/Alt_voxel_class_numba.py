import numpy as np
from numba import njit, prange
from typing import List, Optional, Self

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

        self.x_origin, self.y_origin, self.z_origin = coordinates

        # Einfuehrung einer Toleranz fuer die Modulo-Berechnung
        #tolerance = 1e-6

        '''
        # Ueberpruefen, ob width, depth, height durch voxel_size teilbar sind, unter Beruecksichtigung der Toleranz
        if (abs(width % voxel_size) > tolerance or
            abs(depth % voxel_size) > tolerance or
            abs(height % voxel_size) > tolerance):
            raise ValueError("Width, depth, and height must be divisible by voxel_size within the specified tolerance.")
            '''

        # Berechnung der Anzahl der Voxel in jeder Dimension
        self.x_voxels = int(round(width / voxel_size))
        self.y_voxels = int(round(depth / voxel_size))
        self.z_voxels = int(round(height / voxel_size))

        # Initialisierung des Fill-Status als boolesches Numpy-Array
        self.voxels_fill = np.ones((self.x_voxels, self.y_voxels, self.z_voxels), dtype=np.bool_)

    def get_total_volume(self) -> float:
        """
        Berechnet das gesamte Volumen der gefuellten Voxel.
        """
        return np.sum(self.voxels_fill) * (self.voxel_size ** 3)

    def apply_tool(self, tool: Tool, penetration_depth: float = 0.0) -> None:
        """
        Setzt das Werkzeug auf das Bauteil. Alle Voxel-Elemente, welche das Werkzeug beruehren,
        werden als entfernt identifiziert durch das Setzen von fill auf False.
        :param tool: Ein definiertes Werkzeug aus der Tool-Klasse.
        :param penetration_depth: Die Tiefe, bis zu der das Werkzeug das Bauteil durchdringt. !!!Diese Wert scheint kein Funktion zu haben? Immer auf 1 setzen
        """
        # Berechne die Eindringtiefe entlang der Z-Achse
        tool_z_penetration = tool.z_position

        # Annahme: Das Werkzeug steht senkrecht zum Werkstueck und ist nicht verdreht in x- oder y-Achse
        relative_penetration = tool_z_penetration - self.z_origin
        affected_z_index = int(relative_penetration / self.voxel_size)

        # Berechnung der Anzahl der Z-Schichten, die entfernt werden sollen
        if penetration_depth > 0:
            z_layers_to_remove = int(round(penetration_depth / self.voxel_size))
        else:
            z_layers_to_remove = 1  # Standardmaessig eine Schicht

        # Sicherstellen, dass wir nicht ueber die Grenzen hinausgehen
        end_z_index = min(affected_z_index + z_layers_to_remove, self.z_voxels)

        if 0 <= affected_z_index < self.z_voxels:
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
        else:
            pass
            #print(f"Werkzeugposition {tool_z_penetration} liegt ausserhalb des Bauteils.")

    def apply_tool_test(self, tool: Tool) -> None:
        """
        Setzt das Werkzeug auf das Bauteil. Alle Voxel-Elemente, welche das Werkzeug beruehren und sich auf oder ueber der Werkzeugposition befinden,
        werden als entfernt identifiziert durch das Setzen von fill auf False.
        :param tool: Ein definiertes Werkzeug aus der Tool-Klasse.
        """
        # Berechnung der betroffenen Z-Schichten basierend auf der Werkzeugposition
        # Da das Werkzeug von oben kommt, beeinflusst es alle Schichten mit Z >= tool.z_position
        # Z-Koordinate des Bauteils beginnt bei self.z_origin und steigt nach oben

        # Umrechnung der Z-Koordinate des Werkzeugs in Voxel-Index
        tool_z_index = int(np.floor((tool.z_position - self.z_origin) / self.voxel_size))
        # Alle Schichten ab tool_z_index nach oben werden beeinflusst
        min_z_index = max(0, tool_z_index)
        max_z_index = self.z_voxels - 1  # Oberste Schicht des Bauteils

        # Berechnung der betroffenen x- und y-Bereiche
        min_i = max(0, int(np.floor((tool.x_position - tool.radius - self.x_origin) / self.voxel_size)))
        max_i = min(self.x_voxels - 1, int(np.floor((tool.x_position + tool.radius - self.x_origin) / self.voxel_size)))
        min_j = max(0, int(np.floor((tool.y_position - tool.radius - self.y_origin) / self.voxel_size)))
        max_j = min(self.y_voxels - 1, int(np.floor((tool.y_position + tool.radius - self.y_origin) / self.voxel_size)))

        if min_z_index <= max_z_index:
            for z in range(min_z_index, max_z_index + 1):
                apply_tool_numba(
                    self.voxels_fill[:, :, z],
                    self.x_origin,
                    self.y_origin,
                    self.voxel_size,
                    tool.x_position,
                    tool.y_position,
                    tool.radius,
                    min_i,
                    max_i,
                    min_j,
                    max_j
                )
        else:
            print(f"Werkzeugposition {tool.z_position} liegt unterhalb des Bauteils.")


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

                # Finde den naechsten Punkt auf dem Voxel zum Werkzeugzentrum
                nearest_x = max(voxel_x, min(tool_x, voxel_x + voxel_size))
                nearest_y = max(voxel_y, min(tool_y, voxel_y + voxel_size))

                # Berechne die Quadrat-Distanz
                dist_x = tool_x - nearest_x
                dist_y = tool_y - nearest_y
                dist_sq = dist_x ** 2 + dist_y ** 2

                if dist_sq < radius_sq:
                    fill_layer[i, j] = False

def apply_tool_numba_test(fill_layer, x_origin, y_origin, voxel_size, tool_x, tool_y, tool_radius, min_i, max_i, min_j, max_j):
    """
    Numba-optimierte Funktion zur Anwendung des Werkzeugs auf eine Schicht von Voxel-Fill-Status.
    Setzt den Fill-Status auf False fuer Voxel, die innerhalb des Werkzeugradius liegen.
    """
    radius_sq = tool_radius ** 2

    for i in prange(min_i, max_i + 1):
        for j in range(min_j, max_j + 1):
            if fill_layer[i, j]:
                # Berechne die Koordinaten des Zentrums des aktuellen Voxels
                voxel_x = x_origin + (i + 0.5) * voxel_size
                voxel_y = y_origin + (j + 0.5) * voxel_size

                # Berechne die Quadrat-Distanz zum Werkzeugzentrum
                dist_x = tool_x - voxel_x
                dist_y = tool_y - voxel_y
                dist_sq = dist_x ** 2 + dist_y ** 2

                if dist_sq <= radius_sq:
                    fill_layer[i, j] = False