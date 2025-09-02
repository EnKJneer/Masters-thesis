import numpy as np
import matplotlib.pyplot as plt

class ActivationPlotter:
    def __init__(self):
        # KIT Farbpalette
        self.kit_red = "#D30015"
        self.kit_green = "#009682"
        self.kit_yellow = "#FFFF00"
        self.kit_orange = "#FFC000"
        self.kit_blue = "#0C537E"
        self.kit_dark_blue = "#002D4C"
        self.kit_magenta = "#A3107C"

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    def tanh(self, x):
        return np.tanh(x)

    def add_arrow(self, ax, direction="right"):
        """Fügt einen Pfeil an das Ende der Achse hinzu, der 1% der Achsenlänge übersteht."""
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        if direction == "right":
            x_pos = x_max
            y_pos = 0
            arrow_length = 0.01 * (x_max - x_min)
            ax.annotate('', xy=(x_pos, y_pos), xytext=(x_pos-arrow_length, y_pos),
                        arrowprops=dict(arrowstyle='->', color=self.kit_dark_blue, lw=0.5))
        elif direction == "up":
            x_pos = 0
            y_pos = y_max
            arrow_length = 0.01 * (y_max - y_min)
            ax.annotate('', xy=(x_pos, y_pos), xytext=(x_pos, y_pos-arrow_length),
                        arrowprops=dict(arrowstyle='->', color=self.kit_dark_blue, lw=0.5))

    def create_plots(self, output_path="activation_functions.pdf"):
        x = np.linspace(-10, 10, 1000)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
        fig.suptitle("Aktivierungsfunktionen", fontsize=16, fontweight='bold', color=self.kit_dark_blue)

        for ax in axes.flat:
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(self.kit_dark_blue)
            ax.spines['bottom'].set_color(self.kit_dark_blue)
            ax.tick_params(axis='x', colors=self.kit_dark_blue)
            ax.tick_params(axis='y', colors=self.kit_dark_blue)
            ax.grid(True, color=self.kit_dark_blue, alpha=0.3, linewidth=0.5, linestyle='-')

        # Sigmoid
        axes[0, 0].plot(x, self.sigmoid(x), color=self.kit_green, linewidth=2)
        axes[0, 0].set_title("Sigmoid", color=self.kit_dark_blue, fontsize=12, fontweight='bold')
        axes[0, 0].set_ylim(-0.1, 1.1)
        self.add_arrow(axes[0, 0], direction="right")
        self.add_arrow(axes[0, 0], direction="up")

        # ReLU
        axes[0, 1].plot(x, self.relu(x), color=self.kit_green, linewidth=2)
        axes[0, 1].set_title("ReLU", color=self.kit_dark_blue, fontsize=12, fontweight='bold')
        axes[0, 1].set_ylim(-1, 10)
        self.add_arrow(axes[0, 1], direction="right")
        self.add_arrow(axes[0, 1], direction="up")

        # ELU
        axes[1, 0].plot(x, self.elu(x), color=self.kit_green, linewidth=2)
        axes[1, 0].set_title("ELU", color=self.kit_dark_blue, fontsize=12, fontweight='bold')
        axes[1, 0].set_ylim(-1.1, 10)
        self.add_arrow(axes[1, 0], direction="right")
        self.add_arrow(axes[1, 0], direction="up")

        # Tanh
        axes[1, 1].plot(x, self.tanh(x), color=self.kit_green, linewidth=2)
        axes[1, 1].set_title("Tanh", color=self.kit_dark_blue, fontsize=12, fontweight='bold')
        axes[1, 1].set_ylim(-1.1, 1.1)
        self.add_arrow(axes[1, 1], direction="right")
        self.add_arrow(axes[1, 1], direction="up")

        plt.tight_layout()
        fig.savefig(output_path, format="pdf", bbox_inches='tight')  # Speichern als PDF
        plt.close(fig)
        return output_path

# Beispielaufruf
plotter = ActivationPlotter()
pdf_path = plotter.create_plots("activation_functions.pdf")
print(f"Plot wurde als PDF gespeichert unter: {pdf_path}")
