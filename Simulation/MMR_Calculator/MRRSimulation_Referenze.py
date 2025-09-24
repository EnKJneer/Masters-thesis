import numpy as np


def is_tool_in_part(tool_radius, a_p, pos_x, pos_y, pos_z, part_position, part_dimension):
    """Prüft, ob das Werkzeug vollständig im Werkstück liegt (X/Y mit Radius, Z mit Schnitttiefe)."""
    x_min = pos_x - tool_radius
    x_max = pos_x + tool_radius
    y_min = pos_y - tool_radius
    y_max = pos_y + tool_radius
    z_min = pos_z - a_p /10
    z_max = pos_z + a_p /10  # Werkzeugspitze

    part_x_min = part_position[0]
    part_x_max = part_position[0] + part_dimension[0]
    part_y_min = part_position[1]
    part_y_max = part_position[1] + part_dimension[1]
    part_z_min = part_position[2]
    part_z_max = part_position[2] + part_dimension[2]

    return (
            x_min >= part_x_min and x_max <= part_x_max and
            y_min >= part_y_min and y_max <= part_y_max and
            z_min >= part_z_min and z_max <= part_z_max
    )

def calculateMRR(process_data, part, tool, ap_array, part_position, part_dimension, frequence = 50):

    new_part_volume = part.get_total_volume()
    print(f'Volumen vorher: {new_part_volume}')

    mrr = []

    for i in range(process_data.shape[0]):

        #Ermitteln a_p
        a_p = ap_array[i]

        #Position aus aktuelle Datenpunkt lesen
        pos_x = process_data.loc[i, 'pos_x']
        pos_y = process_data.loc[i, 'pos_y']
        pos_z = process_data.loc[i, 'pos_z']

        new_tool_coordinates = [pos_x, pos_y, pos_z]
        # Geschwindigkeit aus aktuelle Datenpunkt lesen
        v_sp = process_data.loc[i, 'v_sp']

        #Einstellung aktuelle Position von Tool
        tool.set_new_position(new_tool_coordinates)

        #Berechnen von MRR

        if v_sp == 0:
            materialremoved_sim = 0
        elif not is_tool_in_part(tool.radius, a_p, pos_x, pos_y, pos_z, part_position, part_dimension):
            materialremoved_sim = 0
        else:
            #print(voxel_image.find_false_coordinates(part.voxels_fill))             #Z-Achse Koordinate nicht bekannt dann diese Zeile ausfuehren um rauszu finden welche Ebene abgertragen wird!
            old_part_volume = new_part_volume
            part.apply_tool(tool)
            new_part_volume = part.get_total_volume()

            if old_part_volume != new_part_volume:

                materialremoved_sim = (old_part_volume - new_part_volume) * frequence

                print(f'MRR_{i}: {materialremoved_sim}')

            elif old_part_volume < new_part_volume:
                raise ValueError("Alte Volumen ist groesser als neue Volume!")

            else:
                materialremoved_sim = 0

        mrr.append(materialremoved_sim)

    return np.array(mrr)