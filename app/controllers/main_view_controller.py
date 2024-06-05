import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class MainViewController(QObject):
    """Main view controller.

    Handles the main window view. Receives signals from the view and sends
    signals to the model.

    Attributes:
        _model (MLTeachModel): the model
        _view (MainAppView): the main window view
    """
    # controller -> model signals
    propogate_dataset_load_request_signal = pyqtSignal(str)
    propogate_split_data_request_signal = pyqtSignal(str, str)
    propogate_create_model_request_signal = pyqtSignal(str, tuple)
    propogate_fit_model_request_signal = pyqtSignal()
    propogate_preprocess_request_signal = pyqtSignal(str)
    propogate_request_data_signal = pyqtSignal(str)
    propogate_view_predictions_request_signal = pyqtSignal()
    propogate_evaluation_request_signal = pyqtSignal()
    propogate_kfold_req_signal = pyqtSignal(int)

    # controller -> view signals
    propogate_data_load_success_signal = pyqtSignal(str)
    propogate_model_creation_success_signal = pyqtSignal(bool, str)
    propogate_preprocess_success_signal = pyqtSignal(bool)
    propogate_data_split_success_signal = pyqtSignal(bool)
    propogate_send_data_signal = pyqtSignal(str, list, np.ndarray, np.ndarray)
    propogate_model_fit_success_signal = pyqtSignal(bool)
    propogate_predictions_signal = pyqtSignal(np.ndarray, np.ndarray)
    propogate_eval_res_signal = pyqtSignal(str, dict)
    propogate_kfold_res_signal = pyqtSignal(tuple)

    def __init__(self, mlteach_model, main_view) -> None:
        """Initialize MainViewController.

        Args:
            model (MLTeachModel): the model
            view (MainAppView): the main window view
        """
        super().__init__()

        self._model = mlteach_model
        self._view = main_view
        self.set_up_view_controller_connections()
        self.set_up_model_controller_connections()

    def set_up_view_controller_connections(self) -> None:
        """Set up connections between main_view and main_view_controller."""
        # view -> controller
        self._view.dataset_load_request_signal.connect(
            self.on_dataset_load_request
        )
        self._view.split_data_request_signal.connect(
            self.on_split_data_request
        )
        self._view.create_model_request_signal.connect(
            self.on_create_model_request
        )
        self._view.fit_model_request_signal.connect(
            self.on_fit_model_request
        )
        self._view.preprocess_request_signal.connect(
            self.on_preprocess_request_signal
        )
        self._view.request_data_signal.connect(
            self.on_view_data_request
        )
        self._view.request_predictions_signal.connect(
            self.on_view_predictions_request
        )
        self._view.request_evaluation_signal.connect(
            self.on_view_evaluation_request
        )
        self._view.request_kfold_signal.connect(
            self.on_kfold_request
        )

        # controller -> view
        self.propogate_data_load_success_signal.connect(
            self._view.on_data_loaded_update_view
        )
        self.propogate_model_creation_success_signal.connect(
            self._view.on_model_creation_success
        )
        self.propogate_preprocess_success_signal.connect(
            self._view.on_preprocess_success
        )
        self.propogate_data_split_success_signal.connect(
            self._view.on_data_split_success
        )
        self.propogate_send_data_signal.connect(
            self._view.on_data_received
        )
        self.propogate_model_fit_success_signal.connect(
            self._view.on_model_fit_success
        )
        self.propogate_predictions_signal.connect(
            self._view.on_predictions_received
        )
        self.propogate_eval_res_signal.connect(
            self._view.on_eval_res_received
        )
        self.propogate_kfold_res_signal.connect(
            self._view.on_kfold_res_received
        )

    def set_up_model_controller_connections(self) -> None:
        """Set up connections between main_view_controller and mlteach_model."""
        # controller -> model
        self.propogate_dataset_load_request_signal.connect(
            self._model.load_data
        )
        self.propogate_split_data_request_signal.connect(
            self._model.split_data
        )
        self.propogate_create_model_request_signal.connect(
            self._model.create_model
        )
        self.propogate_fit_model_request_signal.connect(
            self._model.fit_model
        )
        self.propogate_preprocess_request_signal.connect(
            self._model.preprocess_data
        )
        self.propogate_request_data_signal.connect(
            self._model.get_data
        )
        self.propogate_view_predictions_request_signal.connect(
            self._model.get_predictions
        )
        self.propogate_evaluation_request_signal.connect(
            self._model.evaluate_model
        )
        self.propogate_kfold_req_signal.connect(
            self._model.run_kfold
        )

        # model -> controller
        self._model.data_load_success_signal.connect(
            self.on_data_load_success
        )
        self._model.data_split_success_signal.connect(
            self.on_data_split_success
        )
        self._model.model_creation_success_signal.connect(
            self.on_model_creation_success
        )
        self._model.preprocess_success_signal.connect(
            self.on_preprocess_success
        )
        self._model.send_data_signal.connect(
            self.on_send_data_signal_received
        )
        self._model.model_fit_success_signal.connect(
            self.on_model_fit_success
        )
        self._model.send_predictions_signal.connect(
            self.on_predictions_received
        )
        self._model.send_eval_res_signal.connect(
            self.on_eval_res_received
        )
        self._model.send_kfold_res_signal.connect(
            self.on_kfold_res_received
        )

    @pyqtSlot(str)
    def on_dataset_load_request(self, file_path: str) -> None:
        """Handle dataset load request.

        Controller -> Model.

        Args:
            file_path (str): path to the dataset file
        """
        self.propogate_dataset_load_request_signal.emit(file_path)

    @pyqtSlot(str)
    def on_preprocess_request_signal(self, method: str) -> None:
        """Handle check data loaded request.

        Controller -> Model.
        """
        self.propogate_preprocess_request_signal.emit(method)

    @pyqtSlot(bool)
    def on_preprocess_success(self, success: bool) -> None:
        """Handle preprocess success signal.

        Model -> Controller.

        Args:
            success (bool): whether the preprocess was successful
        """
        self.propogate_preprocess_success_signal.emit(success)

    @pyqtSlot(str)
    def on_data_load_success(self, file_path: str) -> None:
        """Handle data loaded view update request.

        Controller -> View.

        Args:
            file_path (str): path to the dataset file
        """
        self.propogate_data_load_success_signal.emit(file_path)

    @pyqtSlot(str, str)
    def on_split_data_request(self, seed: str, split: str) -> None:
        """Handle split data request.

        Controller -> Model.

        Args:
            seed (str): the seed value
            split_ratio (str): the split ratio
        """
        self.propogate_split_data_request_signal.emit(seed, split)

    @pyqtSlot(bool)
    def on_data_split_success(self, success: bool) -> None:
        """Handle data split success signal.

        Model -> Controller.

        Args:
            success (bool): whether the data split was successful
        """
        self.propogate_data_split_success_signal.emit(success)

    @pyqtSlot(str)
    def on_view_data_request(self, which_data: str) -> None:
        """Handle view train data request.

        Controller -> View.
        """
        self.propogate_request_data_signal.emit(which_data)

    @pyqtSlot(str, list, np.ndarray, np.ndarray)
    def on_send_data_signal_received(self, which_data: str, headers: list, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Propogate the data to the view.

        Controller -> view.

        Args:
            X_train (np.ndarray): the training features
            y_train (np.ndarray): the training target
        """
        self.propogate_send_data_signal.emit(which_data, headers, X_train, y_train)

    @pyqtSlot(str, tuple)
    def on_create_model_request(self, model: str, params: tuple) -> None:
        """Handle create model request.

        Controller -> Model.

        Args:
            model (str): the model type
            lambda_text (float): the lambda value
        """
        self.propogate_create_model_request_signal.emit(model, params)

    @pyqtSlot(bool, str)
    def on_model_creation_success(self, success: bool, model_type: str) -> None:
        """Handle model creation success signal.

        Model -> Controller.

        Args:
            success (bool): whether the model creation was successful
            model_type (str): the model type
        """
        self.propogate_model_creation_success_signal.emit(success, model_type)

    @pyqtSlot()
    def on_fit_model_request(self) -> None:
        """Handle fit model request.

        Controller -> Model.
        """
        self.propogate_fit_model_request_signal.emit()

    @pyqtSlot(bool)
    def on_model_fit_success(self, success: bool) -> None:
        """Handle model fit success signal.

        Model -> Controller.

        Args:
            success (bool): whether the model fit was successful
        """
        self.propogate_model_fit_success_signal.emit(success)

    @pyqtSlot()
    def on_view_predictions_request(self) -> None:
        """Handle view predictions request.

        Controller -> Model.
        """
        self.propogate_view_predictions_request_signal.emit()

    @pyqtSlot(np.ndarray, np.ndarray)
    def on_predictions_received(self, actual, predictions: np.ndarray) -> None:
        """Handle predictions received signal.

        Model -> Controller.

        Args:
            predictions (np.ndarray): the predictions
        """
        self.propogate_predictions_signal.emit(actual, predictions)

    @pyqtSlot()
    def on_view_evaluation_request(self) -> None:
        """Handle view evaluation request.

        Controller -> Model.
        """
        self.propogate_evaluation_request_signal.emit()

    @pyqtSlot(str, dict)
    def on_eval_res_received(self, model: str, eval_res: dict) -> None:
        """Handle evaluation results received signal.

        Model -> Controller.

        Args:
            eval_res (dict): the evaluation results
        """
        self.propogate_eval_res_signal.emit(model, eval_res)

    @pyqtSlot(int)
    def on_kfold_request(self, k: int) -> None:
        """Handle kfold request.

        Controller -> Model.
        """
        self.propogate_kfold_req_signal.emit(k)

    @pyqtSlot(tuple)
    def on_kfold_res_received(self, res: tuple) -> None:
        """Handle kfold results received signal.

        Model -> Controller.

        Args:
            res (tuple): the kfold results
        """
        self.propogate_kfold_res_signal.emit(res)
