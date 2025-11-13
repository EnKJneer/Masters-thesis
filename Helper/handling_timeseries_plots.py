import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import List, Dict, Optional

# KIT-Farben (zentral definiert)
KIT_RED = "#D30015"
KIT_GREEN = "#009682"
KIT_YELLOW = "#FFFF00"
KIT_ORANGE = "#FFC000"
KIT_BLUE = "#0C537E"
KIT_DARK_BLUE = "#002D4C"
KIT_MAGENTA = "#A3107C"
KIT_GRAY = "#767676"

def setup_plot(
    kit_dark_blue: str,
    line_size: float,
    fontsize_axis_label: int,
) -> tuple:
    """Erstellt die Grundstruktur der Plots (Achsen, Stile, etc.)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Stile für ax_v (Vorschubgeschwindigkeit)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(kit_dark_blue)
    ax.spines['bottom'].set_color(kit_dark_blue)
    ax.spines['left'].set_linewidth(line_size)
    ax.spines['bottom'].set_linewidth(line_size)

    # Plot Vorschubgeschwindigkeit
    ax.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5)
    ax.tick_params(axis='both', colors=kit_dark_blue, labelsize=fontsize_axis_label)

    return fig, ax


def setup_subplots(
    time: np.ndarray,
    v_colname: str,
    data: pd.DataFrame,
    kit_dark_blue: str,
    line_size: float,
    plot_line_size: float,
    kit_blue: str,
    v_label: str,
    fontsize_axis: int,
) -> tuple:
    """Erstellt die Grundstruktur der Plots (Achsen, Stile, etc.)."""
    fig, (ax_v, ax_i) = plt.subplots(
        2, 1, figsize=(14, 12), dpi=300,
        sharex=True, height_ratios=[1, 3],
        gridspec_kw={'hspace': 0.05}
    )

    # Stile für ax_v (Vorschubgeschwindigkeit)
    for ax in [ax_v, ax_i]:
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(kit_dark_blue)
        ax.spines['bottom'].set_color(kit_dark_blue)
        ax.spines['left'].set_linewidth(line_size)
        ax.spines['bottom'].set_linewidth(line_size)

    # Plot Vorschubgeschwindigkeit
    line_v, = ax_v.plot(time, data[v_colname], label=v_label, color=kit_blue, linewidth=plot_line_size)
    ax_v.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5)
    ax_i.grid(True, color=kit_dark_blue, alpha=0.3, linewidth=0.5)
    ax_v.set_axisbelow(True)
    ax_v.tick_params(axis='both', colors=kit_dark_blue, labelsize=fontsize_axis)
    ax_i.tick_params(axis='both', colors=kit_dark_blue, labelsize=fontsize_axis)

    return fig, ax_v, ax_i, line_v

def plot_v_and_i(
    ax_i: plt.Axes,
    time: np.ndarray,
    data: pd.DataFrame,
    col_name: str,
    kit_blue: str,
    plot_line_size: float,
    y_configs: List[Dict[str, str]],
    y_colors: List[str],
) -> List[plt.Line2D]:
    """Plottet Strom-Messwerte und zusätzliche y-Achsen."""
    lines_pred = []


    if y_configs is None:
        y_configs = []

    for i, config in enumerate(y_configs):
        color = y_colors[i % len(y_colors)]
        line, _ = plot_prediction_with_std(data, config['ycolname'], color, config['ylabel'])
        if line is not None:
            lines_pred.append(line)

    line_i, = ax_i.plot(time, data[col_name], label='Strom-Messwerte', color=kit_blue, linewidth=plot_line_size)

    return [line_i] + lines_pred

def add_axis_labels(
    ax_v: plt.Axes,
    ax_i: plt.Axes,
    v_axis: str,
    label: str,
    kit_dark_blue: str,
    fontsize_axis_label: int,
    line_size: float,
) -> None:
    """Fügt Achsenbeschriftungen hinzu."""
    # X-Achsenbeschriftung (ax_v)
    xmin, xmax = ax_v.get_xlim()
    ymin, ymax = ax_v.get_ylim()
    y_pos = -0.2 * ymax
    ax_v.annotate('', xy=(xmax, 0), xytext=(xmax*0.95, 0),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_v.text(xmax*0.95, y_pos, r'$t$ in s', ha='left', va='center', color=kit_dark_blue, fontsize=fontsize_axis_label)

    # Y-Achsenbeschriftung (ax_v)
    x_label_pos_y = -0.06 * (xmax - xmin)
    y_label_pos_y = ymax * 0.6
    ax_v.annotate('', xy=(0, ymax), xytext=(0, ymax - 0.08*(ymax-ymin)),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_v.text(x_label_pos_y, y_label_pos_y - 0.04*(ymax-ymin), v_axis,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=fontsize_axis_label)

    # X-Achsenbeschriftung (ax_i)
    xmin, xmax = ax_i.get_xlim()
    ymin, ymax = ax_i.get_ylim()
    x_pos = -0.08 * (xmax - xmin)
    y_pos = -0.15 * ymax
    arrow_length = 0.04 * (xmax - xmin)
    ax_i.set_xlim(left=min(x_pos, xmin), right=xmax + arrow_length/2)
    ax_i.annotate('', xy=(xmax + arrow_length/2, 0), xytext=(xmax - arrow_length/2, 0),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_i.text(xmax*0.95, y_pos, r'$t$ in s', ha='left', va='center', color=kit_dark_blue, fontsize=fontsize_axis_label)

    # Y-Achsenbeschriftung (ax_i)
    y_pos = ymax * 0.85
    arrow_length = 0.04*(ymax-ymin)
    ax_i.set_ylim(bottom=min(y_pos, ymin), top=ymax + arrow_length / 2)
    ax_i.annotate('', xy=(0, ymax + arrow_length/2), xytext=(0, ymax - arrow_length/2),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_i.text(x_pos, y_pos - 0.04*(ymax-ymin), label,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=fontsize_axis_label)

def add_axes_labels(
    ax_i: plt.Axes,
    label: str,
    kit_dark_blue: str,
    fontsize_axis_label: int,
    line_size: float,
) -> None:
    """Fügt Achsenbeschriftungen hinzu."""
    # X-Achsenbeschriftung (ax_i)
    xmin, xmax = ax_i.get_xlim()
    ymin, ymax = ax_i.get_ylim()
    x_pos = -0.125 * (xmax - xmin)
    y_pos = -0.15 * ymax
    arrow_length = 0.04 * (xmax - xmin)
    ax_i.set_xlim(left=min(x_pos, xmin), right=xmax + arrow_length/2)
    ax_i.annotate('', xy=(xmax + arrow_length/2, 0), xytext=(xmax - arrow_length/2, 0),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_i.text(xmax*0.95, y_pos, r'$t$ in s', ha='left', va='center', color=kit_dark_blue, fontsize=fontsize_axis_label)

    # Y-Achsenbeschriftung (ax_i)
    y_pos = ymax * 0.85
    arrow_length = 0.04*(ymax-ymin)
    ax_i.set_ylim(bottom=min(y_pos, ymin), top=ymax + arrow_length / 2)
    ax_i.annotate('', xy=(0, ymax + arrow_length/2), xytext=(0, ymax - arrow_length/2),
                 arrowprops=dict(arrowstyle='->', color=kit_dark_blue, lw=line_size, mutation_scale=25))
    ax_i.text(x_pos, y_pos - 0.06*(ymax-ymin), label,
             ha='center', va='bottom', color=kit_dark_blue, fontsize=fontsize_axis_label)


def add_legend_and_save(
    fig: plt.Figure,
    ax_i: plt.Axes,
    legend_elements: List,
    legend_labels: List,
    title: str,
    kit_dark_blue: str,
    fontsize_title: int,
    fontsize_axis_label: int,
    filename: str,
    path: str,
    dpi: int,
    data_types:List = ['.svg', '.pdf']
) -> None:
    """Fügt Legende hinzu und speichert den Plot."""

    if len(legend_elements) > 1:
        fig.legend(
            handles=legend_elements,
            labels=legend_labels,
            loc='lower center',
            ncol=2,
            frameon=True,
            facecolor='white',
            edgecolor=kit_dark_blue,
            framealpha=1.0,
            fontsize=fontsize_axis_label,
            bbox_to_anchor=(0.5, -0.1),  # Position unterhalb des Plots
        )

    fig.suptitle(title, color=kit_dark_blue, fontsize=fontsize_title, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    plot_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    for type in data_types:
        fig.savefig(plot_path + type, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved as {plot_path}')

def plot_y_configs(
    ax: plt.Axes,
    data: pd.DataFrame,
    y_configs: List[Dict[str, str]],
    y_colors: List[str],
    time: np.ndarray,
    plot_line_size: float = 2,
) -> List[plt.Line2D]:
    """Plottet zusätzliche y-Konfigurationen (z. B. Vorhersagen oder Abweichungen)."""
    lines_pred = []
    if y_configs is None:
        y_configs = []

    for i, config in enumerate(y_configs):
        color = y_colors[i % len(y_colors)]
        line, _ = plot_prediction_with_std(
            data,
            config['ycolname'],
            color,
            config['ylabel']
        )
        if line is not None:
            lines_pred.append(line)

    return lines_pred

def plot_prediction_with_std(data, base_label, color, label=''):
    """
    Hilfsfunktion zum Plotten von Vorhersagen mit Standardabweichung.
    """
    cols = [col for col in data.columns if col.startswith(base_label)]
    if not cols:
        return None, None

    mean = data[cols].mean(axis=1)
    std = data[cols].std(axis=1)
    line, = plt.gca().plot(data.index / 50, mean, label=label, color=color, linewidth=2)
    plt.gca().fill_between(data.index / 50, mean - std, mean + std, color=color, alpha=0.2)
    return line, mean


def plot_time_series_with_sections(
    data: pd.DataFrame,
    title: str,
    filename: str,
    dpi: int = 300,
    col_name: str = 'curr_x',
    label: str = '$I$\nin A',
    y_configs: List[Dict[str, str]] = None,
    f_a: int = 50,
    path: str = 'Plots',
    v_colname: str = 'v_x',
    v_label: str = 'Vorschubgeschwindigkeit',
    v_axis: str = 'v\nin m/s',
    v_threshold: float = 1.0,
    data_types=['.svg', '.pdf'],
        fontsize_axis: int = 14,
        fontsize_axis_label: int = 16,
        fontsize_title: int = 18,
        add_v_axis: bool = True,
) -> None:
    """Erstellt einen DIN 461 konformen Zeitverlaufsplan (ursprüngliche Variante)."""
    # Konfigurationen
    line_size = 1.5
    plot_line_size = 2
    y_colors = [KIT_RED, KIT_ORANGE, KIT_MAGENTA, KIT_YELLOW, KIT_GREEN]

    # Zeitachse
    time = data.index / f_a

    # Plot aufbauen
    fig, ax_v, ax_i, line_v = setup_subplots(
        time, v_colname, data, KIT_DARK_BLUE, line_size, plot_line_size, KIT_DARK_BLUE, v_label, fontsize_axis_label
    )

    # Bereiche mit |v| < v_threshold markieren
    v_abs = np.abs(data[v_colname])
    low_speed_mask = v_abs < v_threshold
    diff = np.diff(low_speed_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if low_speed_mask.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if low_speed_mask.iloc[-1]:
        ends = np.append(ends, len(low_speed_mask) - 1)

    for start, end in zip(starts, ends):
        ax_i.axvspan(time[start], time[end], color=KIT_GREEN, alpha=0.2, linewidth=0)

    # Strom und zusätzliche y-Achsen plotten
    legend_elements = plot_v_and_i(ax_i, time, data, col_name, KIT_BLUE, plot_line_size, y_configs, y_colors)

    # Achsenbeschriftungen hinzufügen
    add_axis_labels(ax_v, ax_i, v_axis, label, KIT_DARK_BLUE, fontsize_axis_label, line_size)

    # Legende und speichern
    legend_elements = [line_v] + legend_elements
    legend_labels = [line.get_label() for line in legend_elements]
    legend_elements.append(Patch(facecolor=KIT_GREEN, alpha=0.2, label=f'Bereiche mit |v| < {v_threshold} m/s'))
    legend_labels.append(f'|v| < {v_threshold} m/s')

    add_legend_and_save(
        fig, ax_i, legend_elements, legend_labels, title, KIT_DARK_BLUE, fontsize_title, fontsize_axis,
        filename, path, dpi, data_types
    )

def plot_time_series_sections(
        data: pd.DataFrame,
        title: str,
        filename: str,
        dpi: int = 300,
        col_name: str = 'curr_x',
        axis_name: str = '$I$ in A',
        label: str = 'Strom-Messwerte',
        speed_threshold: float = 1.0,
        f_a: int = 50,
        path: str = 'Plots',
        y_configs: List[Dict[str, str]] = None,
        data_types=['.svg', '.pdf'],
        fontsize_axis: int=16,
        fontsize_axis_label: int = 16,
        fontsize_label: int = 14,
        fontsize_title: int = 18,
        line_size: int = 1.5,
        plot_line_size: int = 2,
        v_colname: str = 'v_x',
        v_threshold: float = 1.0,
) -> None:
    """Erstellt einen DIN 461 konformen Zeitverlaufsplan mit:
    - Oberem Plot: Vorschubgeschwindigkeit (eingefärbt nach Vorzeichen der Kraft)
    - Unterem Plot: Strom (eingefärbt nach z für |v| < speed_threshold)
    - Unterstützung für y_configs im unteren Plot
    """
    # Konfigurationen

    y_colors = [KIT_RED, KIT_ORANGE, KIT_MAGENTA, KIT_YELLOW, KIT_GREEN]

    # Zeitachse
    time = data.index / f_a

    # ----- Plot aufbauen (zwei Achsen: oben für Vorschubgeschwindigkeit, unten für Strom) -----
    fig, ax = setup_plot(
        KIT_DARK_BLUE, line_size, fontsize_axis
    )
    # Bereiche mit |v| < v_threshold markieren
    v_abs = np.abs(data[v_colname])
    low_speed_mask = v_abs < v_threshold
    diff = np.diff(low_speed_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if low_speed_mask.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if low_speed_mask.iloc[-1]:
        ends = np.append(ends, len(low_speed_mask) - 1)

    for start, end in zip(starts, ends):
        ax.axvspan(time[start], time[end], color=KIT_GREEN, alpha=0.2, linewidth=0)

    # ----- Plot für Strom-Messwerte und y_configs -----
    lines_pred = plot_y_configs(ax, data, y_configs, y_colors, time, plot_line_size)
    line_i, = ax.plot(time, data[col_name], label=label, color=KIT_BLUE, linewidth=plot_line_size)

    # ----- Achsenbeschriftungen -----
    add_axes_labels(ax, axis_name, KIT_DARK_BLUE, fontsize_axis_label, line_size)

    # ----- Legende -----
    legend_elements = lines_pred + [line_i]
    legend_labels = [line.get_label() for line in legend_elements]

    legend_elements.append(Patch(facecolor=KIT_GREEN, alpha=0.2, label=f'Bereiche mit |v| < {v_threshold} m/s'))
    legend_labels.append(f'|v| < {v_threshold} m/s')

    # ----- Titel, Legende und Speichern -----
    add_legend_and_save(
        fig, ax, legend_elements, legend_labels, title, KIT_DARK_BLUE, fontsize_title, fontsize_axis_label,
        filename, path, dpi, data_types
    )

def plot_time_series_with_sections_force_colored(
    data: pd.DataFrame,
    title: str,
    filename: str,
    dpi: int = 300,
    col_name: str = 'curr_x',
    label: str = '$I$\nin A',
    y_configs: List[Dict[str, str]] = None,
    f_a: int = 50,
    path: str = 'Plots',
    v_colname: str = 'v_x',
    v_label: str = 'Vorschubgeschwindigkeit',
    v_axis: str = 'v in m/s',
    v_threshold: float = 1.0,
    force_colname: str = 'F_x',
    data_types = ['.svg', '.pdf'],
        fontsize_axis: int = 14,
        fontsize_axis_label: int = 16,
        fontsize_title: int = 18
) -> None:
    """Erstellt einen DIN 461 konformen Zeitverlaufsplan mit Kraft-Einfärbung (neue Variante)."""
    # Konfigurationen
    line_size = 1.5
    plot_line_size = 2
    y_colors = [KIT_RED, KIT_ORANGE, KIT_MAGENTA, KIT_YELLOW, KIT_GREEN]

    # Zeitachse
    time = data.index / f_a

    # Plot aufbauen
    fig, ax_v, ax_i, line_v = setup_subplots(
        time, v_colname, data, KIT_DARK_BLUE, line_size, plot_line_size, KIT_DARK_BLUE, v_label, fontsize_axis_label
    )

    # Einfärben des oberen Plots nach Vorzeichen der Kraft
    force_sign = np.sign(data[force_colname])
    diff = np.diff(force_sign)
    starts = np.where(diff != 0)[0] + 1  # Indizes der Vorzeichenwechsel + 1
    starts = np.insert(starts, 0, 0)  # Füge den Startindex 0 hinzu
    ends = np.concatenate((starts[1:], [len(force_sign) - 1]))  # Ende des letzten Abschnitts

    for start, end in zip(starts, ends):
        if force_sign[start] > 0:
            ax_v.axvspan(time[start], time[end], color=KIT_RED, alpha=0.25, linewidth=0)
        elif force_sign[start] < 0:
            ax_v.axvspan(time[start], time[end], color=KIT_ORANGE, alpha=0.25, linewidth=0)

    # Bereiche mit |v| < v_threshold markieren (unterer Plot)
    v_abs = np.abs(data[v_colname])
    low_speed_mask = v_abs < v_threshold
    diff = np.diff(low_speed_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if low_speed_mask.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if low_speed_mask.iloc[-1]:
        ends = np.append(ends, len(low_speed_mask) - 1)

    for start, end in zip(starts, ends):
        ax_i.axvspan(time[start], time[end], color=KIT_GREEN, alpha=0.2, linewidth=0)

    # Strom und zusätzliche y-Achsen plotten
    legend_elements = plot_v_and_i(ax_i, time, data, col_name, KIT_BLUE, plot_line_size, y_configs, y_colors)

    # Achsenbeschriftungen hinzufügen
    add_axis_labels(ax_v, ax_i, v_axis, label, KIT_DARK_BLUE, fontsize_axis_label, line_size)

    # Legende und speichern
    legend_elements = [line_v] + legend_elements
    legend_labels = [line.get_label() for line in legend_elements]
    legend_elements.extend([
        Patch(facecolor=KIT_GREEN, alpha=0.2, label=f'Bereiche mit |v| < {v_threshold} m/s'),
        Patch(facecolor=KIT_RED, alpha=0.25, label='Positive Kraft'),
        Patch(facecolor=KIT_ORANGE, alpha=0.25, label='Negative Kraft'),
    ])
    legend_labels.extend([
        f'|v| < {v_threshold} m/s',
        'Positive Kraft',
        'Negative Kraft',
    ])

    add_legend_and_save(
        fig, ax_i, legend_elements, legend_labels, title, KIT_DARK_BLUE, fontsize_title, fontsize_axis,
        filename, path, dpi, data_types
    )

def plot_time_series_error_with_sections(
    data: pd.DataFrame,
    title: str,
    filename: str,
    dpi: int = 300,
    col_name: str = 'curr_x',
    label: str = '$I$\nin A',
    y_configs: List[Dict[str, str]] = None,
    f_a: int = 50,
    path: str = 'Plots',
    v_colname: str = 'v_x',
    v_label: str = 'Vorschubgeschwindigkeit',
    v_axis: str = 'v in m/s',
    v_threshold: float = 1.0,
    fontsize_axis: int =14,
    fontsize_axis_label: int  = 16,
    fontsize_title: int  = 18
) -> None:
    """Erstellt einen DIN 461 konformen Zeitverlaufsplan mit Fehlerdarstellung auf zwei y-Achsen."""
    # Konfigurationen
    line_size = 1.5
    plot_line_size = 2
    y_colors = [KIT_RED, KIT_ORANGE, KIT_MAGENTA, KIT_YELLOW, KIT_GREEN]

    # Zeitachse
    time = data.index / f_a

    # Plot aufbauen
    fig, ax_v, ax_i, line_v = setup_subplots(
        time, v_colname, data, KIT_DARK_BLUE, line_size, plot_line_size, KIT_BLUE, v_label, fontsize_axis_label
    )

    # Zweite y-Achse für Abweichungen erstellen
    ax_i2 = ax_i.twinx()
    ax_i2.spines['bottom'].set_visible(False)
    ax_i2.spines['top'].set_visible(False)
    ax_i2.spines['left'].set_visible(False)
    ax_i2.spines['right'].set_color(KIT_RED)
    ax_i2.spines['right'].set_linewidth(line_size)

    # Bereiche mit |v| < v_threshold markieren (unterer Plot)
    v_abs = np.abs(data[v_colname])
    low_speed_mask = v_abs < v_threshold
    diff = np.diff(low_speed_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if low_speed_mask.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if low_speed_mask.iloc[-1]:
        ends = np.append(ends, len(low_speed_mask) - 1)
    for start, end in zip(starts, ends):
        ax_i.axvspan(time[start], time[end], color=KIT_GREEN, alpha=0.2, linewidth=0)

    # Plot für Strom-Messwerte auf erster Achse
    line_i, = ax_i.plot(time, data[col_name], label='Strom-Messwerte', color=KIT_BLUE, linewidth=plot_line_size, zorder=2)

    # Skalierungsfaktor berechnen
    y_min_i, y_max_i = ax_i.get_ylim()
    y_min_i2 = y_max_i
    y_max_i2 = y_min_i
    # Plot für Abweichungen (Mittelwert der Vorhersagen - tatsächlicher Strom) auf zweiter Achse
    lines_pred = []
    if y_configs is None:
        y_configs = []
    for i, config in enumerate(y_configs):
        color = y_colors[i % len(y_colors)]
        cols = [col for col in data.columns if col.startswith(config['ycolname'])]
        if not cols:
            continue
        mean_pred = data[cols].mean(axis=1)
        deviation = mean_pred - data[col_name]

        # Skalierungsfaktor berechnen
        y_min_i2 = min(y_min_i2, min(deviation))
        y_max_i2 = max(y_max_i2, max(deviation))

        # Skalierung der Abweichung
        scale_factor = (y_max_i - y_min_i) / (y_max_i2 - y_min_i2)

        # Abweichung  plotten
        scaled_deviation = deviation  * scale_factor/2 #y_max_i + (deviation - y_max_i2) * scale_factor
        line, = ax_i.plot(
            time, scaled_deviation,
            label=f'Abweichung {config["ylabel"]}',
            color=color, linewidth=plot_line_size, zorder=1
        )
        lines_pred.append(line)

    # Achsenbeschriftungen hinzufügen
    add_axis_labels(ax_v, ax_i, v_axis, label, KIT_DARK_BLUE, fontsize_axis_label, line_size)
    #ax_i2.set_ylabel('Abweichung', color=KIT_DARK_BLUE, fontsize=fontsize_axis_label)
    ax_i2.tick_params(axis='y', labelcolor=KIT_RED, labelsize=fontsize_axis_label)

    # Achsenlimits von ax_i2 anpassen (optional)
    scale_factor = (y_max_i - y_min_i) / (y_max_i2 - y_min_i2)
    offset = y_min_i2 - y_min_i / scale_factor
    ax_i2.set_ylim(y_min_i2-offset, y_max_i2-offset)

    # Y-Achsenbeschriftung (ax_i)
    xmin, xmax = ax_i2.get_xlim()
    ymin, ymax = ax_i2.get_ylim()
    y_pos = ymax * 0.85
    arrow_length = 0.04*(ymax-ymin)
    ax_i2.set_ylim(bottom=min(y_pos, ymin), top=ymax + arrow_length / 2)
    ax_i2.annotate('', xy=(xmax, ymax + arrow_length/2), xytext=(xmax, ymax - arrow_length/2),
                 arrowprops=dict(arrowstyle='->', color=KIT_RED, lw=line_size, mutation_scale=25))
    ax_i2.text(xmax*0.95, y_pos - 0.04*(ymax-ymin), label,
             ha='center', va='bottom', color=KIT_RED, fontsize=fontsize_axis_label)

    # Legende und speichern
    legend_elements = [line_i, line_v] + lines_pred
    legend_labels = [line.get_label() for line in legend_elements]
    legend_elements.append(Patch(facecolor=KIT_GREEN, alpha=0.2, label=f'Bereiche mit |v| < {v_threshold} m/s'))
    legend_labels.append(f'|v| < {v_threshold} m/s')
    add_legend_and_save(
        fig, ax_i, legend_elements, legend_labels, title, KIT_DARK_BLUE, fontsize_title, fontsize_axis_label,
        filename, path, dpi
    )


def plot_time_series_with_sections_colored(
    data: pd.DataFrame,
    title: str,
    filename: str,
    dpi: int = 300,
    col_name: str = 'curr_x',
    label: str = 'Strom in A',
    v_colname: str = 'v_x',
    v_label: str = 'Vorschubgeschwindigkeit',
    v_axis: str = 'v in m/s',
    force_colname: str = 'F_x',  # Spalte für die Kraft (Vorzeichen)
    z_col: str = 'z',  # Spalte für z (unterer Plot)
    speed_threshold: float = 1.0,
    f_a: int = 50,
    path: str = 'Plots',
    lane1: float = None,
    lane2: float = None,
    y_configs: List[Dict[str, str]] = None,
    data_types = ['.svg', '.pdf']
) -> None:
    """Erstellt einen DIN 461 konformen Zeitverlaufsplan mit:
    - Oberem Plot: Vorschubgeschwindigkeit (eingefärbt nach Vorzeichen der Kraft)
    - Unterem Plot: Strom (eingefärbt nach z für |v| < speed_threshold)
    - Unterstützung für y_configs im unteren Plot
    """
    # Konfigurationen
    fontsize_axis = 16
    fontsize_axis_label = 16
    fontsize_label = 14
    fontsize_title = 18
    line_size = 1.5
    plot_line_size = 2
    y_colors = [KIT_RED, KIT_ORANGE, KIT_MAGENTA, KIT_YELLOW, KIT_GREEN]

    # Zeitachse
    time = data.index / f_a

    # ----- Plot aufbauen (zwei Achsen: oben für Vorschubgeschwindigkeit, unten für Strom) -----
    fig, ax_v, ax_i, line_v = setup_subplots(
        time, v_colname, data, KIT_DARK_BLUE, line_size, plot_line_size, KIT_DARK_BLUE, v_label, fontsize_axis_label
    )

    # ----- Einfärben des oberen Plots nach Vorzeichen der Kraft -----
    force_sign = np.sign(data[force_colname])
    diff = np.diff(force_sign)
    starts = np.where(diff != 0)[0] + 1
    starts = np.insert(starts, 0, 0)  # Erster Abschnitt
    ends = np.concatenate((starts[1:], [len(force_sign) - 1]))

    for start, end in zip(starts, ends):
        if force_sign[start] > 0:
            ax_v.axvspan(time[start], time[end], color=KIT_RED, alpha=0.25, linewidth=0)
        elif force_sign[start] < 0:
            ax_v.axvspan(time[start], time[end], color=KIT_ORANGE, alpha=0.25, linewidth=0)

    # ----- Unterer Plot: Einfärben nach z für |v| < speed_threshold -----
    v_abs = np.abs(data[v_colname])
    low_speed_mask = v_abs < speed_threshold
    diff = np.diff(low_speed_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    if low_speed_mask.iloc[0]:
        starts = np.insert(starts, 0, 0)
    if low_speed_mask.iloc[-1]:
        ends = np.append(ends, len(low_speed_mask) - 1)

    for start, end in zip(starts, ends):
        z_segment = data[z_col].iloc[start:end+1]
        z_mean = np.mean(z_segment)
        color = KIT_RED if z_mean > 0 else KIT_ORANGE
        ax_i.axvspan(time[start], time[end], color=color, alpha=0.15, linewidth=0)

    # ----- Plot für Strom-Messwerte und y_configs -----
    lines_pred = plot_y_configs(ax_i, data, y_configs, y_colors, time, plot_line_size)
    line_i, = ax_i.plot(time, data[col_name], label='Strom-Messwerte', color=KIT_BLUE, linewidth=plot_line_size)

    # ----- Gestrichelte Näherungslinien (+/-lane1 und +/-lane2) -----
    if lane1 is not None:
        ax_i.axhline(y=lane1, color=KIT_GREEN, linestyle='--', linewidth=2, label=f'Näherung +{lane1} A')
        ax_i.axhline(y=-lane1, color=KIT_GREEN, linestyle='--', linewidth=2, label=f'Näherung -{lane1} A')
    if lane2 is not None:
        ax_i.axhline(y=lane2, color=KIT_MAGENTA, linestyle='--', linewidth=2, label=f'Näherung +{lane2} A')
        ax_i.axhline(y=-lane2, color=KIT_MAGENTA, linestyle='--', linewidth=2, label=f'Näherung -{lane2} A')

    # ----- Achsenbeschriftungen -----
    add_axis_labels(ax_v, ax_i, v_axis, label, KIT_DARK_BLUE, fontsize_axis_label, line_size)

    # ----- Legende -----
    legend_elements = lines_pred + [line_i, line_v]
    legend_labels = [line.get_label() for line in legend_elements]
    legend_elements.extend([
        Patch(facecolor=KIT_RED, alpha=0.25, label='Positive Kraft'),
        Patch(facecolor=KIT_ORANGE, alpha=0.25, label='Negative Kraft'),
        Patch(facecolor=KIT_RED, alpha=0.15, label=f'Bereiche mit |v| < {speed_threshold} m/s und z > 0'),
        Patch(facecolor=KIT_ORANGE, alpha=0.15, label=f'Bereiche mit |v| < {speed_threshold} m/s und z ≤ 0')
    ])
    legend_labels.extend([
        'Positive Kraft',
        'Negative Kraft',
        f'|v| < {speed_threshold} m/s, z > 0',
        f'|v| < {speed_threshold} m/s, z ≤ 0'
    ])
    if lane1 is not None:
        legend_elements.extend([
            Line2D([0], [0], color=KIT_GREEN, linestyle='--', label=f'Näherung ±{lane1} A'),
        ])
        legend_labels.extend([
            f'±{lane1} A',
        ])
    if lane2 is not None:
        legend_elements.extend([
            Line2D([0], [0], color=KIT_GREEN, linestyle='--', label=f'Näherung ±{lane2} A'),
        ])
        legend_labels.extend([
            f'±{lane2} A',
        ])

    # ----- Titel, Legende und Speichern -----
    add_legend_and_save(
        fig, ax_i, legend_elements, legend_labels, title, KIT_DARK_BLUE, fontsize_title, fontsize_axis,
        filename, path, dpi, data_types
    )

def plot_time_series(
        data: pd.DataFrame,
        title: str,
        filename: str,
        dpi: int = 300,
        col_name: str = 'curr_x',
        axis_name: str = '$I$ in A',
        label: str = 'Strom-Messwerte',
        speed_threshold: float = 1.0,
        f_a: int = 50,
        path: str = 'Plots',
        y_configs: List[Dict[str, str]] = None,
        data_types=['.svg', '.pdf'],
        fontsize_axis: int=16,
        fontsize_axis_label: int = 16,
        fontsize_label: int = 14,
        fontsize_title: int = 18,
        line_size: int = 1.5,
        plot_line_size: int = 2
) -> None:
    """Erstellt einen DIN 461 konformen Zeitverlaufsplan mit:
    - Oberem Plot: Vorschubgeschwindigkeit (eingefärbt nach Vorzeichen der Kraft)
    - Unterem Plot: Strom (eingefärbt nach z für |v| < speed_threshold)
    - Unterstützung für y_configs im unteren Plot
    """
    # Konfigurationen

    y_colors = [KIT_RED, KIT_ORANGE, KIT_MAGENTA, KIT_YELLOW, KIT_GREEN]

    # Zeitachse
    time = data.index / f_a

    # ----- Plot aufbauen (zwei Achsen: oben für Vorschubgeschwindigkeit, unten für Strom) -----
    fig, ax = setup_plot(
        KIT_DARK_BLUE, line_size, fontsize_axis
    )

    # ----- Plot für Strom-Messwerte und y_configs -----
    lines_pred = plot_y_configs(ax, data, y_configs, y_colors, time, plot_line_size)
    line_i, = ax.plot(time, data[col_name], label=label, color=KIT_BLUE, linewidth=plot_line_size)

    # ----- Achsenbeschriftungen -----
    add_axes_labels(ax, axis_name, KIT_DARK_BLUE, fontsize_axis_label, line_size)

    # ----- Legende -----
    legend_elements = lines_pred + [line_i]
    legend_labels = [line.get_label() for line in legend_elements]

    legend_labels.extend([
        'Positive Kraft',
        'Negative Kraft',
        f'|v| < {speed_threshold} m/s, z > 0',
        f'|v| < {speed_threshold} m/s, z ≤ 0'
    ])

    # ----- Titel, Legende und Speichern -----
    add_legend_and_save(
        fig, ax, legend_elements, legend_labels, title, KIT_DARK_BLUE, fontsize_title, fontsize_axis,
        filename, path, dpi, data_types
    )

def plot_time_series_with_variance(data_list, title, dpi=300, col_name='curr_x', label_axis='$I$ in A', f_a=50,
                                   filename=None, path='Plots',    fontsize_axis_label = 14,
    fontsize_title = 16,
    line_size = 1.5,
    plot_line_size = 2):
    """
    Erstellt einen DIN 461 konformen Zeitverlaufsplot mit Mittelwert, Standardabweichung und Originalkurven.
    :param data_list: Liste von DataFrames (jeweils eine Version)
    :param title: Titel des Plots
    :param dpi: Auflösung des Plots
    :param col_name: Spaltenname der zu plottenden Daten (z.B. 'curr_x')
    :param label_axis: Beschriftung der y-Achse
    :param f_a: Abtastrate für die Zeitachse
    :param filename: Dateiname zum Speichern
    :param path: Pfad zum Speichern
    """
    # Konfigurationen

    # Minimale Länge der Datenreihen bestimmen
    min_length = min(len(df[col_name]) for df in data_list)
    # Daten auf minimale Länge kürzen
    truncated_data = [df[col_name][:min_length] for df in data_list]
    # Zeitachse erstellen
    time = np.arange(min_length) / f_a
    # Mittelwert und Standardabweichung berechnen
    mean_values = np.mean(truncated_data, axis=0)
    std_values = np.std(truncated_data, axis=0)
    # Plot aufbauen
    fig, ax = setup_plot(KIT_DARK_BLUE, line_size, fontsize_axis_label)
    # Originalkurven (transparent) plotten
    colors = [KIT_RED, KIT_ORANGE, KIT_MAGENTA]
    lines_original = []
    for i, values in enumerate(truncated_data):
        line, = ax.plot(time, values, color=colors[i], alpha=0.3, linewidth=1, label=f'Version {i + 1}')
        lines_original.append(line)
    # Mittelwert plotten
    line_mean, = ax.plot(time, mean_values, label='Mittelwert', color=KIT_BLUE, linewidth=plot_line_size)
    # Standardabweichung als Schatten darstellen
    ax.fill_between(time, mean_values - std_values, mean_values + std_values,
                    color=KIT_BLUE, alpha=0.2, label='±1 Std.-Abw.')
    # Achsenbeschriftungen hinzufügen
    add_axes_labels(ax, label_axis, KIT_DARK_BLUE, fontsize_axis_label, line_size)
    # Legende und speichern
    legend_elements = lines_original + [line_mean, Patch(facecolor=KIT_BLUE, alpha=0.2, label='±1 Std.-Abw.')]
    legend_labels = [line.get_label() for line in lines_original] + ['Mittelwert', '±1 Std.-Abw.']
    add_legend_and_save(fig, ax, legend_elements, legend_labels, title, KIT_DARK_BLUE, fontsize_title,
                        fontsize_axis_label, filename, path, dpi)