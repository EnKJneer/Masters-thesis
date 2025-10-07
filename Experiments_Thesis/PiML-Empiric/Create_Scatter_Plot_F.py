from collections import deque

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def plot_scatter_fx_vs_curr(
    data: pd.DataFrame,
    title: str,
    filename: str,
    dpi: int = 300,
    curr_col: str = 'curr_x',
    fx_col: str = 'f_x_sim',
    v_col: str = 'v_x',
    z_col: str = 'z',
    v_threshold: float = 10.0,
    path: str = 'Plots',
) -> None:
    """
    Erstellt einen DIN-konformen Scatter-Plot von f_x_sim vs. curr_x mit:
    - Filterung: Nur Punkte mit |v_x| ≤ v_threshold
    - Farbkodierung: kit_green für z > 0, kit_blue für z ≤ 0
    - KIT-Farben, Achsenbeschriftung, Legende
    """
    # KIT-Farben (aus deiner Vorlage)
    kit_red = "#D30015"
    kit_green = "#009682"
    kit_blue = "#0C537E"
    kit_dark_blue = "#002D4C"
    kit_gray = "#767676"

    # Daten filtern
    mask = np.abs(data[v_col]) <= v_threshold
    data_filtered = data[mask].copy()

    # Farben basierend auf z > 0
    colors = np.where(data_filtered[z_col] > 0, kit_green, kit_blue)

    # Plot erstellen
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    # Scatter-Plot
    scatter = ax.scatter(
        data_filtered[curr_col],
        data_filtered[fx_col],
        c=colors,
        alpha=0.6,
        edgecolors='none',
        s=20,
        label='Datenpunkte'
    )

    # Achsen-Stil (DIN-konform)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(kit_dark_blue)
    ax.spines['bottom'].set_color(kit_dark_blue)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # Grid und Ticks
    ax.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', colors=kit_dark_blue)

    # Achsenbeschriftung (manuell, wie in deiner Vorlage)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # X-Achse (curr_x)
    ax.annotate('', xy=(xmax, 0), xytext=(xmax*0.95, 0),
                arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5))
    ax.text(xmax*0.95, -0.07*ymax, r'$I$ in A',
            ha='left', va='center', color=kit_dark_blue, fontsize=12)

    # Y-Achse (f_x_sim)
    ax.annotate('', xy=(0, ymax), xytext=(0, ymax*0.95),
                arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=1.5))
    ax.text(-0.06*(xmax-xmin), ymax*0.85, r'$F$ in N',
            ha='center', va='bottom', color=kit_dark_blue, fontsize=12)

    # Titel
    fig.suptitle(title, color=kit_dark_blue, fontsize=14, fontweight='bold', y=0.95)

    # Legende
    legend_elements = [
        Patch(facecolor=kit_green, label='z > 0'),
        Patch(facecolor=kit_blue, label='z ≤ 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=kit_gray,
               markersize=10, label=f'Gefiltert: |v_x| ≤ {v_threshold}')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=1.0)

    # Speichern
    import os
    os.makedirs(path, exist_ok=True)
    plot_path = os.path.join(path, filename)
    fig.savefig(plot_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(plot_path + '.pdf', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Gespeichert unter: {plot_path}')

def sign_hold(v, eps=1e-1, n=3, init=-1):
    # Initialisierung des Arrays z mit Nullen
    z = np.zeros(len(v))
    h_init = np.ones(n) * init

    assert n > 1

    # Initialisierung des FiFo h mit Länge 5 und Initialwerten 0
    h = deque(h_init, maxlen=n)

    # Berechnung von z
    for i in range(len(v)):
        if abs(v[i]) > eps:
            h.append(v[i])

        if i >= n - 1:  # Da wir ab dem 5. Element starten wollen
            # Berechne zi als Vorzeichen der Summe
            z[i] = np.sign(sum(h))

    return z

# Beispielaufruf
if __name__ == '__main__':
    # Daten laden
    df = pd.read_csv('Results/EmpiricModel-2025_10_07_09_23_22/Predictions/DMC60H_S235JR_Plate_Normal_3.csv')

    df['z'] = sign_hold(df['v_x'])

    plot_scatter_fx_vs_curr(
        data=df,
        title='Prozesskraft zum Motorstrom',
        filename='scatter_fx_vs_curr_filtered',
        v_threshold=10.0,
        dpi=600
    )
