import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def find_max_z_with_data(array):
    """
    Findet die groesste Z-Ebene, die mindestens einen True-Wert enthaelt.

    Parameters:
    - array (np.ndarray): 3D NumPy-Array mit binaeren Werten (dtype=bool).

    Returns:
    - max_z (int): Der groesste Z-Index mit mindestens einem True-Wert.
    - None: Wenn keine Z-Ebene True-Werte enthaelt.
    """
    # Iteriere von der hoechsten Z-Ebene zurueck
    for z in reversed(range(array.shape[2])):
        if np.any(array[:, :, z]):
            return z
    return None

def find_false_coordinates(array):
    """
    Findet die Koordinaten aller Elemente in einem 3D-Array, die den Wert False haben.
    
    Parameters:
    - array (np.ndarray): 3D NumPy-Array mit booleschen Werten.
    
    Returns:
    - coordinates (list of tuples): Liste von Koordinaten (x, y, z) mit False-Werten.
    """
    if array.ndim != 3:
        raise ValueError("Das Array muss 3-dimensional sein.")
    
    # Verwende np.where, um die Indizes der False-Werte zu finden
    false_indices = np.where(~array)
    
    # Zippe die Indizes zu einer Liste von (x, y, z) Tupeln
    coordinates = list(zip(false_indices[0], false_indices[1], false_indices[2]))
    
    return coordinates

def array_to_image(i: int, voxel_fill, z: int, save_path=None, show_image=True) -> None:
    """
    Konvertiert eine 3D-binaere NumPy-Array (dtype=bool) in ein Schwarz-Weiss-Bild fuer eine feste Z-Ebene.

    Parameters:
    - array (np.ndarray): 3D NumPy-Array mit binaeren Werten (dtype=bool).
    - z (int): Die Z-Ebene, die extrahiert werden soll.
    - save_path (str, optional): Pfad zum Speichern des Bildes. Wenn None, wird das Bild nicht gespeichert.
    - show_image (bool, optional): Wenn True, wird das Bild angezeigt.

    """
    # Ueberpruefen, ob das Array 3D ist
    if voxel_fill.ndim != 3:
        raise ValueError("Das Array muss 3-dimensional sein.")
    
    # Ueberpruefen, ob z innerhalb der gueltigen Bereiche liegt
    if z < 0 or z >= voxel_fill.shape[2]:
        z = voxel_fill.shape[2] - 1
        #raise ValueError(f"Z-Wert muss zwischen 0 und {voxel_fill.shape[2]-1} liegen.")
    
    # Extrahiere die 2D-Scheibe bei der gegebenen Z-Ebene
    slice_2d = voxel_fill[:, :, z]
    #print (slice_2d)
    
    # Konvertiere boolesche Werte zu uint8 (0 und 255)
    img_array = slice_2d.astype(np.uint8) * 255
    
    # Erstellen eines PIL Image-Objekts im Graustufenmodus
    img = Image.fromarray(img_array, mode='L')
    
    # Bild speichern, falls ein Pfad angegeben ist
    if save_path:
        img.save(f"{save_path}_{i}.png ", format='PNG')
        #print(f"Bild gespeichert unter: {save_path}")
    
    # Bild anzeigen, falls gewuenscht
    if show_image:
        plt.imshow(img, cmap='gray')
        plt.title(f"Z = {z}_{i}")
        plt.axis('off')  # Achsen ausschalten
        plt.show()
    
    return None
