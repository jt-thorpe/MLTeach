from datetime import datetime

import numpy as np
from app.resources.widgets.data_viewer import DataViewer
from app.views.knowledge_base_view import KnowledgeBaseView
from app.views.main_view_ui import Ui_MainWindow
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox

# Error message constants
INVALID_LAMBDA_ERR = "Invalid Lambda"
INVALID_LAMBDA_MSG = "Lambda must be a float."
INVALID_DEGREE_ERR = "Invalid Degree"
INVALID_DEGREE_MSG = "Degree must be an integer."
INVALID_SIGMA_ERR = "Invalid Sigma"
INVALID_SIGMA_MSG = "Sigma must be a float."


class MainView(QMainWindow):
    """Main application view."""
    dataset_load_request_signal = pyqtSignal(str)
    split_data_request_signal = pyqtSignal(str, str)
    create_model_request_signal = pyqtSignal(str, tuple)
    fit_model_request_signal = pyqtSignal()
    preprocess_request_signal = pyqtSignal(str)
    request_data_signal = pyqtSignal(str)
    request_predictions_signal = pyqtSignal()
    request_evaluation_signal = pyqtSignal()
    request_kfold_signal = pyqtSignal(int)

    def __init__(self):
        """Initialize MainView."""
        super().__init__()

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        # Connect menu bar actions
        self._ui.actionLoad_Dataset.triggered.connect(
            self.load_dataset_action)
        self._ui.actionLoad_Knowledge_Base.triggered.connect(
            self.open_knowledge_base_action)

        # Connect buttons to methods
        self._ui.split_data_btn.clicked.connect(self.split_data_btn_clicked)
        self._ui.create_model_btn.clicked.connect(self.create_model_btn_clicked)
        self._ui.fit_btn.clicked.connect(self.fit_btn_clicked)
        self._ui.process_btn.clicked.connect(self.process_btn_clicked)
        self._ui.view_train_btn.clicked.connect(self.view_train_btn_clicked)
        self._ui.view_test_btn.clicked.connect(self.view_test_btn_clicked)
        self._ui.view_scaled_train_btn.clicked.connect(
            self.view_scaled_train_btn_clicked)
        self._ui.view_scaled_test_btn.clicked.connect(
            self.view_scaled_test_btn_clicked)
        self._ui.predict_btn.clicked.connect(self.predict_btn_clicked)
        self._ui.evaluate_btn.clicked.connect(self.evaluate_btn_clicked)
        self._ui.k_fold_btn.clicked.connect(self.kfold_btn_clicked)

        # Disable buttons requiring data to be loaded
        self.disable_buttons()

    def disable_buttons(self):
        """Disable buttons."""
        # Preprocess buttons
        self._ui.split_data_btn.setEnabled(False)
        self._ui.process_btn.setEnabled(False)
        self._ui.view_test_btn.setEnabled(False)
        self._ui.view_train_btn.setEnabled(False)
        self._ui.view_scaled_test_btn.setEnabled(False)
        self._ui.view_scaled_train_btn.setEnabled(False)

        # Model buttons
        self._ui.create_model_btn.setEnabled(False)
        self._ui.fit_btn.setEnabled(False)
        self._ui.predict_btn.setEnabled(False)
        self._ui.k_fold_btn.setEnabled(False)
        self._ui.evaluate_btn.setEnabled(False)

    def load_dataset_action(self):
        """Load dataset action in the menu bar.

        Opens a file dialog to select a dataset file. Emits signal to load dataset.
        """
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open Dataset")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("CSV files (*.csv);;All files (*.*)")

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            self.dataset_load_request_signal.emit(file_path)
            self._ui.console_output.append(
                f"{datetime.now().strftime('%H:%M:%S')} Dataset loaded: {file_path}")

    def open_knowledge_base_action(self):
        """Open the knowledge base view from the menu bar."""
        self.knowledge_base_window = KnowledgeBaseView()
        self.knowledge_base_window.show()

    def split_data_btn_clicked(self):
        """Split data button clicked.

        Get seed and split value, then emit signal to split data.
        """
        seed = self._ui.seed_line_edit.text()
        split = self._ui.split_line_edit.text()

        # If both empty str, or castable to int and float, emit signal
        if len(seed) == 0 and len(split) == 0:
            self._ui.console_output.append(
                f"{datetime.now().strftime('%H:%M:%S')} Splitting data: seed=None, split=default")
            self.split_data_request_signal.emit(seed, split)
            return

        try:
            seed = int(seed) if len(seed) > 0 else 'None'
            split = float(split) if len(split) > 0 else 'None'

            if split != 'None' and (split <= 0 or split >= 1):
                raise ValueError

            self._ui.console_output.append(
                f"{datetime.now().strftime('%H:%M:%S')} Splitting data: seed={seed}, split={split}")
            self.split_data_request_signal.emit(str(seed), str(split))
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Seed or Split", "Seed must be an integer, and split must be a float between 0-1, or both must be empty.")
            return

    @pyqtSlot(bool)
    def on_data_split_success(self, success: bool):
        """Handle data split success signal.

        Args:
            success (bool): whether the data split was successful
        """
        if success:
            self._ui.process_btn.setEnabled(True)
            self._ui.view_train_btn.setEnabled(True)
            self._ui.view_test_btn.setEnabled(True)
            self._ui.create_model_btn.setEnabled(True)
            self._ui.console_output.append(
                f"{datetime.now().strftime('%H:%M:%S')} Data split successful")
        else:
            self._ui.console_output.append(
                f"{datetime.now().strftime('%H:%M:%S')} Data split failed")
            QMessageBox.warning(
                self, "Data Split Failed", "Check your csv file is valid and split value appropriate for the dataset."
            )

    def process_btn_clicked(self):
        """Process button clicked.

        Get preprocess type and emit signal to preprocess data.
        """
        self._ui.console_output.append(
            f"{datetime.now().strftime('%H:%M:%S')} Applying {self._ui.preproc_type_dropdown.currentText()}...")
        self.preprocess_request_signal.emit(self._ui.preproc_type_dropdown.currentText())

    @pyqtSlot(bool)
    def on_preprocess_success(self, success: bool):
        """Handle preprocess success signal.

        Args:
            success (bool): whether the preprocess was successful
        """
        if success:
            self._ui.console_output.append(
                f"{datetime.now().strftime('%H:%M:%S')} {self._ui.preproc_type_dropdown.currentText()} applied")
            self._ui.view_scaled_train_btn.setEnabled(True)
            self._ui.view_scaled_test_btn.setEnabled(True)
        else:
            self._ui.console_output.append(
                f"{datetime.now().strftime('%H:%M:%S')} {self._ui.preproc_type_dropdown.currentText()} failed")

    def view_train_btn_clicked(self):
        """View training data button clicked."""
        self.request_data_signal.emit('train')

    def view_test_btn_clicked(self):
        """View testing data button clicked."""
        self.request_data_signal.emit('test')

    def view_scaled_train_btn_clicked(self):
        """View scaled training data button clicked."""
        self.request_data_signal.emit('train_scaled')

    def view_scaled_test_btn_clicked(self):
        """View scaled testing data button clicked."""
        self.request_data_signal.emit('test_scaled')

    @pyqtSlot(str, list, np.ndarray, np.ndarray)
    def on_data_received(self, which_data: str, headers: list, X: np.ndarray, y: np.ndarray):
        """Handle view training data signal."""
        self._ui.console_output.append(
            f"{datetime.now().strftime('%H:%M:%S')} Data viewer opened")

        # Create data viewer to display data
        self.data_viewer = DataViewer(which_data)
        self.data_viewer.load_data(headers, X, y)
        self.data_viewer.show()

    def create_model_btn_clicked(self):
        """Create model button clicked.

        Get model type and parameters, then emit signal to create model.
        """
        create_methods = {
            0: self._create_ridge_regressor,
            1: self._create_linear_kernel,
            2: self._create_poly_kernel,
            3: self._create_rbf_kernel
        }
        model = self._ui.model_type_dropdown.currentText()
        create_method = create_methods[self._ui.model_type_dropdown.currentIndex()]
        params = create_method()

        if params is None:
            return

        self._ui.console_output.append(
            f"{datetime.now().strftime('%H:%M:%S')} Creating model...")
        self.create_model_request_signal.emit(model, params)

    def _create_ridge_regressor(self):
        """Create params for Ridge Regressor.

        Create params for Ridge Regressor to be used in model creation. Emit signal to create model.

        Returns:
            tuple: (lambda, None)
        """
        lambda_text = self._ui.lambda_line_edit.text()

        if len(lambda_text) == 0:  # If empty, set to default
            lambda_ = 1.0
        else:
            try:
                lambda_ = float(lambda_text)
            except ValueError:
                QMessageBox.warning(
                    self, INVALID_LAMBDA_ERR, INVALID_LAMBDA_MSG)
                return None

        return (lambda_, None)

    def _create_linear_kernel(self):
        """Create params for Linear Kernel.

        Create params for Linear Kernel to be used in model creation. Emit signal to create model.

        Returns:
            tuple: (lambda, None)
        """
        lambda_text = self._ui.lambda_line_edit.text()

        if len(lambda_text) == 0:
            lambda_ = 1.0
        else:
            try:
                lambda_ = float(lambda_text)
            except ValueError:
                QMessageBox.warning(
                    self, INVALID_LAMBDA_ERR, INVALID_LAMBDA_MSG)
                return

        return (lambda_, None)

    def _create_poly_kernel(self):
        """Create params for Polynomial Kernel.

        Create params for Polynomial Kernel to be used in model creation. Emit signal to create model.

        Returns:
            tuple: (lambda, degree)
        """
        lambda_text = self._ui.lambda_line_edit.text()

        if len(lambda_text) == 0:
            lambda_ = 1.0
        else:
            try:
                lambda_ = float(lambda_text)
            except ValueError:
                QMessageBox.warning(
                    self, INVALID_LAMBDA_ERR, INVALID_LAMBDA_MSG)
                return

        degree_text = self._ui.degree_line_edit.text()

        if len(degree_text) == 0:
            degree = 3
        else:
            try:
                degree = int(degree_text)
            except ValueError:
                QMessageBox.warning(
                    self, INVALID_DEGREE_ERR, INVALID_DEGREE_MSG)
                return

        return (lambda_, degree)

    def _create_rbf_kernel(self):
        """Create params for RBF Kernel.

        Create params for RBF Kernel to be used in model creation. Emit signal to create model.

        Returns:
            tuple: (lambda, sigma)
        """
        lambda_text = self._ui.lambda_line_edit.text()

        if len(lambda_text) == 0:
            lambda_ = 1.0
        else:
            try:
                lambda_ = float(lambda_text)
            except ValueError:
                QMessageBox.warning(
                    self, INVALID_LAMBDA_ERR, INVALID_LAMBDA_MSG)
                return

        sigma_text = self._ui.sigma_line_edit.text()

        if len(sigma_text) == 0:
            sigma = 1.0
        else:
            try:
                sigma = float(sigma_text)
            except ValueError:
                QMessageBox.warning(
                    self, INVALID_SIGMA_ERR, INVALID_SIGMA_MSG)
                return

        return (lambda_, sigma)

    def fit_btn_clicked(self):
        """Fit button clicked."""
        self._ui.console_output.append(
            f"{datetime.now().strftime('%H:%M:%S')} Fitting model...")
        self.fit_model_request_signal.emit()

    @pyqtSlot(str)
    def on_data_loaded_update_view(self, file_path: str):
        """Handle data loaded view update request.

        Args:
            file_path (str): path to the dataset file
        """
        self._ui.current_dataset_textedit.setText(file_path)

        # With data loaded, enable
        self._ui.split_data_btn.setEnabled(True)

    @pyqtSlot(bool, str)
    def on_model_creation_success(self, success: bool, model_str: str):
        """Handle model creation success signal."""
        if success:
            self._ui.console_output.append(
                f"{datetime.now().strftime('%H:%M:%S')} Model created: {model_str}")
            self._ui.fit_btn.setEnabled(True)

    @pyqtSlot(bool)
    def on_model_fit_success(self, success: bool):
        """Handle model fit success signal."""
        if success:
            self._ui.console_output.append(
                f"{datetime.now().strftime('%H:%M:%S')} Model fit successful")
            self._ui.predict_btn.setEnabled(True)
            self._ui.k_fold_btn.setEnabled(True)
        else:
            self._ui.console_output.append(
                f"{datetime.now().strftime('%H:%M:%S')} Model fit failed")

    def predict_btn_clicked(self):
        """Predict button clicked."""
        self._ui.console_output.append(
            f"{datetime.now().strftime('%H:%M:%S')} Calculating predictions...")
        self.request_predictions_signal.emit()

    @pyqtSlot(np.ndarray, np.ndarray)
    def on_predictions_received(self, actual, predictions: np.ndarray):
        """Handle predictions received signal."""
        self._ui.console_output.append(
            f"{datetime.now().strftime('%H:%M:%S')} Predictions calculated")

        self._ui.evaluate_btn.setEnabled(True)

        # Display predictions in a data viewer window
        self.pred_viewer = DataViewer('predictions')
        self.pred_viewer.load_predictions(predictions)
        self.pred_viewer.show()

        # Display plots in the visuals display
        self._ui.visuals_display.scatter_plot(actual, predictions)
        self._ui.visuals_display.residual_plot(actual, predictions)

    def evaluate_btn_clicked(self):
        """Evaluate button clicked."""
        self._ui.console_output.append(
            f"{datetime.now().strftime('%H:%M:%S')} Evaluating model...")
        self.request_evaluation_signal.emit()

    @pyqtSlot(str, dict)
    def on_eval_res_received(self, model: str, eval_res: dict):
        """Handle evaluation results received signal."""
        self._ui.console_output.append(
            f"{datetime.now().strftime('%H:%M:%S')} Evaluation complete\n"
            + "-----------\nResults\n-----------\n"
            + model + "\n"
            + f"R-squared: {eval_res['r_squared']}\n" + f"MSE: {eval_res['mean_squared_error']}\n"
            + f"MAE: {eval_res['mean_absolute_error']}\n")

    def kfold_btn_clicked(self):
        """K-Fold button clicked."""
        self._ui.console_output.append(
            f"{datetime.now().strftime('%H:%M:%S')} Attempting K-Fold Cross Validation...")

        if len(self._ui.k_fold_line_edit.text()) == 0:  # Default to 5 if empty
            k = 5
        else:
            try:
                k = int(self._ui.k_fold_line_edit.text())
            except ValueError:
                QMessageBox.warning(
                    self, "Invalid K", "K must be an integer.")
                return

        self.request_kfold_signal.emit(k)

    @pyqtSlot(tuple)
    def on_kfold_res_received(self, kfold_res: tuple):
        """Handle K-Fold results received signal.

        Outputs K-Fold results to the console output.

        Args:
            kfold_res (tuple): (scores, mean_r_squared)
        """
        self._ui.console_output.append(
            f"{datetime.now().strftime('%H:%M:%S')} K-Fold Cross Validation complete\n"
            + "-----------\nResults\n-----------\n"
            + f"Scores: {kfold_res[0]}\n"
            + f"Mean R-squared: {kfold_res[1]}\n")
