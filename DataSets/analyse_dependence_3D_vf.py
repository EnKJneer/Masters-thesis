import os
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

matplotlib.use("TkAgg")

# Verbesserte Grafikqualität
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 9


# Define the linear function for fitting
def linear_func(x, a, b):
    return a * x + b


# Define colors and labels for each file
file_colors = {
    'AL_2007_T4_Plate_Depth_1.csv': ('lightgreen', 'AL Plate Depth 1'),
    'AL_2007_T4_Gear_Depth_1.csv': ('mediumblue', 'AL Gear Depth 1'),
    'AL_2007_T4_Plate_Depth_2.csv': ('darkgreen', 'AL Plate Depth 2'),
    'AL_2007_T4_Gear_Depth_2.csv': ('navy', 'AL Gear Depth 2'),
    'AL_2007_T4_Plate_Depth_3.csv': ('olivedrab', 'AL Plate Depth 3'),
    'AL_2007_T4_Gear_Depth_3.csv': ('darkblue', 'AL Gear Depth 3'),
    'S235JR_Plate_Depth_1.csv': ('sandybrown', 'S Plate Depth 1'),
    'S235JR_Gear_Depth_1.csv': ('lightgray', 'S Gear Depth 1'),
    'S235JR_Plate_Depth_2.csv': ('brown', 'S Plate Depth 2'),
    'S235JR_Gear_Depth_2.csv': ('gray', 'S Gear Depth 2'),
    'S235JR_Plate_Depth_3.csv': ('saddlebrown', 'S Plate Depth 3'),
    'S235JR_Gear_Depth_3.csv': ('darkgray', 'S Gear Depth 3'),
    'AL_2007_T4_Plate_Normal_1.csv': ('blue', 'AL Plate 1'),
    'AL_2007_T4_Gear_Normal_1.csv': ('green', 'AL Gear 1'),
    'AL_2007_T4_Plate_Normal_2.csv': ('blue', 'AL Plate 2'),
    'AL_2007_T4_Gear_Normal_2.csv': ('green', 'AL Gear 2'),
    'AL_2007_T4_Plate_Normal_3.csv': ('blue', 'AL Plate 3'),
    'AL_2007_T4_Gear_Normal_3.csv': ('green', 'AL Gear 3'),
    'AL_2007_T4_Plate_SF_1.csv': ('cyan', 'AL Plate SF 1'),
    'AL_2007_T4_Gear_SF_1.csv': ('teal', 'AL Gear SF 1'),
    'AL_2007_T4_Plate_SF_2.csv': ('cyan', 'AL Plate SF 2'),
    'AL_2007_T4_Gear_SF_2.csv': ('teal', 'AL Gear SF 2'),
    'AL_2007_T4_Plate_SF_3.csv': ('cyan', 'AL Plate SF 3'),
    'AL_2007_T4_Gear_SF_3.csv': ('teal', 'AL Gear SF 3'),
    'S235JR_Gear_Normal_1.csv': ('orange', 'S Gear 1'),
    'S235JR_Plate_Normal_1.csv': ('red', 'S Plate 1'),
    'S235JR_Plate_Normal_2.csv': ('orange', 'S Plate 2'),
    'S235JR_Gear_Normal_2.csv': ('red', 'S Gear 2'),
    'S235JR_Plate_Normal_3.csv': ('orange', 'S Plate 3'),
    'S235JR_Gear_Normal_3.csv': ('red', 'S Gear 3'),
    'S235JR_Plate_SF_1.csv': ('pink', 'S Plate SF 1'),
    'S235JR_Gear_SF_1.csv': ('purple', 'S Gear SF 1'),
    'S235JR_Plate_SF_2.csv': ('pink', 'S Plate SF 2'),
    'S235JR_Gear_SF_2.csv': ('purple', 'S Gear SF 2'),
    'S235JR_Plate_SF_3.csv': ('pink', 'S Plate SF 3'),
    'S235JR_Gear_SF_3.csv': ('purple', 'S Gear SF 3'),
}

path_data = 'DataFiltered'
files = [
    'AL_2007_T4_Plate_Normal_2.csv', 'AL_2007_T4_Gear_Normal_2.csv',
    'AL_2007_T4_Plate_Depth_3.csv', 'AL_2007_T4_Gear_Depth_3.csv',
    #'AL_2007_T4_Plate_SF_2.csv', 'AL_2007_T4_Gear_SF_2.csv',
    #'S235JR_Plate_Normal_2.csv', 'S235JR_Gear_Normal_2.csv',
]

n = 25
axes = ['x']


def sign_hold(v, eps=1e-1):
    z = np.zeros(len(v))
    h = deque([1, 1, 1, 1, 1], maxlen=5)
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])
        if i >= 4:
            z[i] = np.sign(sum(h))
    return z


class Interactive3DPlot:
    def __init__(self, fig, ax):
        self.quiver_objects = {}
        self.fig = fig
        self.ax = ax
        self.scatter_objects = {}
        self.visibility_state = {}
        self.legend_handles = []
        self.legend_labels = []
        self.legend = None

    def add_scatter(self, file, x_data, y_data, z_data, color, label,
                    draw_arrows=True):
        """Fügt einen Scatter-Plot hinzu"""
        scatter = self.ax.scatter(x_data, y_data, z_data, c=color, s=4, alpha=0.8,
                                  label=label, edgecolors='none')
        self.scatter_objects[file] = scatter
        self.visibility_state[file] = True

        # Pfeile hinzufügen, falls gewünscht
        if draw_arrows:
            dx = []
            dy = []
            dz = []
            for i in range(len(x_data) - 1):
                dx.append(x_data[i + 1] - x_data[i])
                dy.append(y_data[i + 1] - y_data[i])
                dz.append(z_data[i + 1] - z_data[i])
            # Startpunkte: x_data[:-1], Endpunkte: x_data[1:]
            quiver = self.ax.quiver(x_data[:-1], y_data[:-1], z_data[:-1],
                                    dx, dy, dz, color=color, length=0.2,
                              alpha=0.5, normalize=True)
            self.quiver_objects[file] = quiver

        # Für die Legende - erstelle einen Proxy Artist
        proxy = plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=color, markersize=10,
                           markeredgecolor='black', markeredgewidth=0.5)
        self.legend_handles.append(proxy)
        self.legend_labels.append(label)

    def toggle_visibility(self, file_index):
        """Schaltet die Sichtbarkeit eines Datensatzes um"""
        file = list(self.scatter_objects.keys())[file_index]
        if file in self.scatter_objects:
            current_state = self.visibility_state[file]
            self.scatter_objects[file].set_visible(not current_state)
            self.quiver_objects[file].set_visible(not current_state)
            self.visibility_state[file] = not current_state
            self.update_legend_appearance()
            self.fig.canvas.draw_idle()
            return not current_state

    def toggle_all(self):
        """Schaltet alle Datensätze an/aus"""
        all_visible = all(self.visibility_state.values())
        new_state = not all_visible
        for file in self.scatter_objects.keys():
            self.scatter_objects[file].set_visible(new_state)
            self.quiver_objects[file].set_visible(new_state)
            self.visibility_state[file] = new_state
        self.update_legend_appearance()
        self.fig.canvas.draw_idle()
        return new_state

    def update_legend_appearance(self):
        """Aktualisiert das Aussehen der Legende"""
        if self.legend:
            for i, file in enumerate(self.scatter_objects.keys()):
                if i < len(self.legend.get_texts()):
                    if self.visibility_state[file]:
                        self.legend.get_texts()[i].set_alpha(1.0)
                        self.legend_handles[i].set_alpha(1.0)
                    else:
                        self.legend.get_texts()[i].set_alpha(0.4)
                        self.legend_handles[i].set_alpha(0.4)
        self.legend.set_draggable(True)  # optional: bessere Usability
        self.fig.canvas.draw_idle()

    def setup_legend_and_controls(self):
        """Erstellt die Legende und Event-Handler"""
        # Erstelle die Legende
        self.legend = self.fig.legend(self.legend_handles, self.legend_labels,
                                      loc='center left', bbox_to_anchor=(1.02, 0.5),
                                      fontsize=9, frameon=True, fancybox=True,
                                      shadow=True, framealpha=0.9)

        # Event-Handler für Tastatureingaben
        def on_key_press(event):
            if event.key in ['1', '2', '3', '4']:
                index = int(event.key) - 1
                if index < len(self.scatter_objects):
                    new_state = self.toggle_visibility(index)
                    label = self.legend_labels[index]
                    print(f"→ {label}: {'EIN' if new_state else 'AUS'}")

            elif event.key == 'a':
                new_state = self.toggle_all()
                print(f"→ Alle Datensätze: {'EIN' if new_state else 'AUS'}")

            elif event.key == 'r':
                # Reset view
                self.ax.view_init(elev=20, azim=45)
                self.fig.canvas.draw_idle()
                print("→ Ansicht zurückgesetzt")

        # Event-Handler für Mausklicks auf Legende
        def on_legend_click(event):
            if self.legend and event.inaxes is None:
                # Prüfe ob Klick in der Legende war
                legend_bbox = self.legend.get_window_extent()
                if legend_bbox.contains(event.x, event.y):
                    # Finde den geklickten Eintrag
                    for i, (text, handle) in enumerate(zip(self.legend.get_texts(), self.legend.legend_handles)):
                        text_bbox = text.get_window_extent()
                        handle_bbox = handle.get_window_extent()

                        if (text_bbox.contains(event.x, event.y) or
                                handle_bbox.contains(event.x, event.y)):
                            if i < len(self.scatter_objects):
                                new_state = self.toggle_visibility(i)
                                label = self.legend_labels[i]
                                print(f"→ {label}: {'EIN' if new_state else 'AUS'}")
                                break

        # Verbinde Event-Handler
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
        self.fig.canvas.mpl_connect('button_press_event', on_legend_click)

        def on_scroll(event):
            if event.button == 'up':
                ax._dist -= 1  # Zoomen rein
            elif event.button == 'down':
                ax._dist += 1  # Zoomen raus
            fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('scroll_event', on_scroll)

        # Fokus auf die Figure setzen für Tastatureingaben
        #self.fig.canvas.focus_set()

        return self.legend


# Hauptschleife für jeden Achsentyp
for axis in axes:
    # Erstelle Figure mit hoher Qualität
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')

    # Erstelle 3D-Plot mit mehr Platz für die Legende
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0.05, right=0.75, top=0.95, bottom=0.05)

    # Erstelle Interactive3DPlot-Objekt
    interactive_plot = Interactive3DPlot(fig, ax)

    # Verarbeite jede Datei
    processed_files = 0
    for file in files:
        try:
            data = pd.read_csv(f'{path_data}/{file}')

            # Datenverarbeitung
            a_axis = data[f'a_{axis}'].iloc[:-n].copy()
            v_axis = data[f'v_{axis}'].iloc[:-n].copy()
            f_axis = data[f'f_{axis}_sim'].iloc[:-n].copy()
            curr_axis = -data[f'curr_{axis}'].iloc[:-n].copy()
            mrr = data['materialremoved_sim'].iloc[:-n].copy()
            time = data.index[:-n].copy()

            # Filtere die Daten
            mask = (np.abs(v_axis) <= 1000) & (~np.isnan(v_axis)) & (~np.isnan(f_axis)) & (~np.isnan(curr_axis))
            v_axis = v_axis[mask]
            f_axis = f_axis[mask]
            curr_axis = curr_axis[mask]

            # Glättung der Daten
            if len(v_axis) > 50:
                d = pd.Series(v_axis)
                v_axis = np.array(d.rolling(10, min_periods=1).mean())
                d = pd.Series(curr_axis)
                curr_axis = np.array(d.rolling(10, min_periods=1).mean())

            # Entferne NaN-Werte nach Glättung
            final_mask = ~(np.isnan(v_axis) | np.isnan(f_axis) | np.isnan(curr_axis))
            v_axis = v_axis[final_mask]
            f_axis = f_axis[final_mask]
            curr_axis = curr_axis[final_mask]

            if len(v_axis) > 0:
                color, label = file_colors.get(file, ('black', file))
                interactive_plot.add_scatter(file, f_axis, v_axis, curr_axis, color, label)
                processed_files += 1
                print(f"✓ Geladen: {file} ({len(v_axis)} Datenpunkte)")
            else:
                print(f"⚠ Leer: {file}")

        except Exception as e:
            print(f"✗ Fehler bei {file}: {e}")
            continue

    if processed_files == 0:
        print("Keine Daten geladen!")
        continue

    # Konfiguriere die Achsen
    ax.set_xlabel(f'Kraft f_{axis} [N]', fontsize=11, labelpad=10)
    ax.set_ylabel(f'Geschwindigkeit v_{axis} [mm/s]', fontsize=11, labelpad=10)
    ax.set_zlabel(f'Strom curr_{axis} [A]', fontsize=11, labelpad=10)
    ax.set_title(f'3D-Visualisierung: Strom vs. Kraft und Geschwindigkeit (Achse {axis})',
                 fontsize=14, pad=20, weight='bold')

    # Verbessere das Aussehen
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Setze Pane-Farben
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    # Erstelle Legende und Controls
    legend = interactive_plot.setup_legend_and_controls()

    # Füge Bedienungsanleitung hinzu
    instruction_text = (
        "🎮 BEDIENUNG:\n"
        "• Tasten 1-4: Datensätze umschalten\n"
        "• Taste 'a': Alle an/aus\n"
        "• Taste 'r': Ansicht zurücksetzen\n"
        "• Maus: Legende anklicken\n"
        "• Maus: Drehen & Zoomen im Plot"
    )

    fig.text(0.77, 0.02, instruction_text, fontsize=9, ha='left', va='bottom',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    # Setze optimale Ansicht
    ax.view_init(elev=20, azim=45)

    # Aktiviere Interaktivität
    ax.mouse_init()

    print(f"\n🎯 Plot erstellt mit {processed_files} Datensätzen")
    print("🎮 Klicken Sie in den Plot und verwenden Sie die Tastaturkürzel!")


plt.tight_layout()
plt.show()

# Erweiterte Bedienungsanleitung
print("\n" + "=" * 70)
print("🎮 ERWEITERTE BEDIENUNGSANLEITUNG")
print("=" * 70)
print("TASTATUR-SHORTCUTS:")
print("  1, 2, 3, 4  → Einzelne Datensätze umschalten")
print("  a           → Alle Datensätze an/aus")
print("  r           → Ansicht zurücksetzen")
print("")
print("MAUS-BEDIENUNG:")
print("  Linke Maustaste + Ziehen    → 3D-Ansicht rotieren")
print("  Rechte Maustaste + Ziehen   → Zoomen")
print("  Mittlere Maustaste + Ziehen → Verschieben")
print("  Klick auf Legende          → Datensatz umschalten")
print("")
print("TIPPS:")
print("  • Klicken Sie zuerst in den Plot-Bereich für Tastatureingaben")
print("  • Status-Meldungen erscheinen in dieser Konsole")
print("  • Ausgegr. Legendeneinträge = ausgeschaltete Datensätze")
print("=" * 70)