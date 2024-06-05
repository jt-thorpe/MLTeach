import sys

from app.controllers.main_view_controller import MainViewController
from app.models.mlteach_model import MLTeachModel
from app.views.main_view import MainView
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QApplication, QStyleFactory

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))

    # Initialize the model, views, and controller
    mlteach_model = MLTeachModel()
    main_view = MainView()
    main_view.setWindowTitle("ML Teach")
    main_view_controller = MainViewController(mlteach_model, main_view)
    main_view.show()

    sys.exit(app.exec())
