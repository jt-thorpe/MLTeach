from app.resources.widgets.residual_plot import ResidualPlotWidget
from app.resources.widgets.scatter_plot import ScatterPlotWidget
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class VisualsDisplay(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)

        self.layout.addWidget(self.scroll_content)

    def add_item(self, item):
        """Add item to the scroll area."""
        self.layout.addWidget(item)

    def residual_plot(self, y, y_pred):
        """Plot actual vs predicted values."""
        plot = ResidualPlotWidget(y, y_pred, parent=self)
        self.add_item(plot)

    def scatter_plot(self, y, y_pred):
        """Plot residuals."""
        plot = ScatterPlotWidget(y, y_pred, parent=self)
        self.add_item(plot)
