from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class ScatterPlotWidget(QWidget):
    def __init__(self, y, y_pred, parent=None):
        super().__init__(parent)

        self.figure = Figure(figsize=(9, 7))
        self.canvas = FigureCanvas(self.figure)

        ax = self.figure.add_subplot(111)
        ax.scatter(y_pred, y, color='blue', alpha=0.6)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
        ax.set_title('Scatter Plot of y vs y_pred')
        ax.set_xlabel('Predicted Values (y_pred)')
        ax.set_ylabel('Actual Values (y)')
        ax.grid(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.setMinimumSize(500, 450)
