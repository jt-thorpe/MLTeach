import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)


class DataViewer(QMainWindow):
    def __init__(self, which_data: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{which_data}")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.table_widget = QTableWidget()
        self.table_widget.verticalHeader().setVisible(False)
        self.layout.addWidget(self.table_widget)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        self.layout.addWidget(self.close_button)

    def load_data(self, headers: list, X: np.ndarray, y: np.ndarray):
        num_samples, num_features = X.shape

        self.table_widget.setRowCount(num_samples)
        self.table_widget.setColumnCount(num_features + 1)  # +1 for target column

        self.table_widget.setHorizontalHeaderLabels(headers)

        for i in range(num_samples):
            for j in range(num_features):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(X[i, j])))
            self.table_widget.setItem(i, num_features, QTableWidgetItem(str(y[i])))

    def load_predictions(self, predictions: np.ndarray):
        num_samples = predictions.shape[0]

        self.table_widget.setRowCount(num_samples)
        self.table_widget.setColumnCount(1)

        self.table_widget.setHorizontalHeaderLabels(["Predictions"])

        for i in range(num_samples):
            self.table_widget.setItem(i, 0, QTableWidgetItem(str(predictions[i])))
