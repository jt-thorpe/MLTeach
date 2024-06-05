import numpy as np
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class ResidualPlotWidget(QWidget):
    def __init__(self, y, y_pred, parent=None):
        super().__init__(parent)

        self.figure = Figure(figsize=(9, 7))
        self.canvas = FigureCanvas(self.figure)

        residuals = np.array(y) - np.array(y_pred)

        ax = self.figure.add_subplot(111)
        ax.scatter(y_pred, residuals, color='red', alpha=0.6)
        ax.axhline(y=0, color='black', linestyle='--')
        ax.set_title('Residual Plot')
        ax.set_xlabel('Predicted Values (y_pred)')
        ax.set_ylabel('Residuals')
        ax.grid(True)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.setMinimumSize(500, 500)
