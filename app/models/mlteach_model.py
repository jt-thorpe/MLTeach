import numpy as np
import pandas as pd
from ml.evaluation.cross_validation import KFold
from ml.models.kernel_ridge_regressor import (KernelRidgeRegressor,
                                              LinearKernelRidgeRegressor,
                                              PolynomialKernelRidgeRegressor,
                                              RbfKernelRidgeRegressor)
from ml.models.ridge_regressor import RidgeRegressor
from ml.preprocessng.standardisation import Standardisation
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from sklearn.model_selection import train_test_split


class MLTeachModel(QObject):
    """MLTeach model.

    Main model for the MLTeach application.
    """
    model_creation_success_signal = pyqtSignal(bool, str)
    data_load_success_signal = pyqtSignal(str)
    data_split_success_signal = pyqtSignal(bool)
    preprocess_success_signal = pyqtSignal(bool)
    send_data_signal = pyqtSignal(str, list, np.ndarray, np.ndarray)
    model_fit_success_signal = pyqtSignal(bool)
    send_predictions_signal = pyqtSignal(np.ndarray, np.ndarray)
    send_eval_res_signal = pyqtSignal(str, dict)
    send_kfold_res_signal = pyqtSignal(tuple)

    def __init__(self) -> None:
        """Initialize MLTeach model."""
        super().__init__()

        self.dataset_file_path: str = None
        # The orignal data
        self.data_orig = {
            "headers": None,
            "features": None,
            "target": None,
        }
        # The data after splitting
        self.data_split = {
            "X_train": None,
            "X_test": None,
            "y_train": None,
            "y_test": None,
        }
        # The data after preprocessing
        self.data_scaled = {
            "X_train": None,
            "X_test": None,
            "y_train": None,
            "y_test": None,
        }
        self.preproc_model = None
        self.ml_model = None
        self.predictions = None

    @pyqtSlot(str)
    def load_data(self, file_path: str) -> None:
        """Read the csv at the given file path and store the data.

        Extracts the headers, features, and target from the csv file as numpy
        arrays and stores them in the data attribute.

        Args:
            file_path (str): path to the data file
        """
        # Save the file path
        self.loaded_data_path = file_path

        # Read csv file, convert to numpy
        data_frame = None
        try:
            data_frame = pd.read_csv(file_path, header=0)
        except Exception as e:
            raise IOError(f"Error reading csv file: {e}")

        self.data_orig["headers"] = data_frame.columns.to_list()
        self.data_orig["features"] = data_frame.iloc[:, :-1].to_numpy()
        self.data_orig["target"] = data_frame.iloc[:, -1].to_numpy()

        # Emit signal to update view after data is loaded
        self.data_load_success_signal.emit(file_path)

    @pyqtSlot(str, str)
    def split_data(self, seed: str, split: str) -> None:
        """Split the data into training and testing sets.

        Args:
            seed (str): the seed for the random split
            split (str): the ratio to split the data
        """
        try:
            # View has already validated the input, but csv could be bad
            if seed == "" and split == "":
                X_train, X_test, y_train, y_test = train_test_split(
                    self.data_orig["features"],
                    self.data_orig["target"],
                )
            elif isinstance(seed, str) and split == "None":
                X_train, X_test, y_train, y_test = train_test_split(
                    self.data_orig["features"],
                    self.data_orig["target"],
                    random_state=int(seed),
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    self.data_orig["features"],
                    self.data_orig["target"],
                    test_size=float(split),
                    random_state=int(seed),
                )

            self.data_split["X_train"] = X_train
            self.data_split["X_test"] = X_test
            self.data_split["y_train"] = y_train
            self.data_split["y_test"] = y_test
            self.data_split_success_signal.emit(True)
        except Exception as e:
            self.data_split_success_signal.emit(False)
            raise ValueError(f"Error splitting data: {e}")

    @pyqtSlot(str)
    def preprocess_data(self, method: str) -> None:
        """Preprocess the data.

        Args:
            method (str): the preprocessing method
        """
        if method == "Standardisation":
            self.preproc_model = Standardisation()
            self.data_scaled["X_train"] = self.preproc_model.fit_transform(self.data_split["X_train"])
            self.data_scaled["X_test"] = self.preproc_model.transform(self.data_split["X_test"])
            self.data_scaled["y_train"] = self.data_split["y_train"]
            self.data_scaled["y_test"] = self.data_split["y_test"]
        elif method == "MinMax":
            pass

        self.preprocess_success_signal.emit(True)

    @pyqtSlot(str)
    def get_data(self, which_data: str) -> None:
        """Get the training data.

        Args:
            which_data (str): the data the user requested
        """
        data_mapping = {
            "train": (self.data_split["X_train"], self.data_split["y_train"]),
            "test": (self.data_split["X_test"], self.data_split["y_test"]),
            "train_scaled": (self.data_scaled["X_train"], self.data_scaled["y_train"]),
            "test_scaled": (self.data_scaled["X_test"], self.data_scaled["y_test"])
        }

        headers = self.data_orig["headers"]
        X, y = data_mapping.get(which_data)
        self.send_data_signal.emit(which_data, headers, X, y)

    @pyqtSlot(str, tuple)
    def create_model(self, model: str, params: tuple) -> None:
        """Create a model.

        Args:
            model (str): the model type
            params (tuple): (lambda_, None) or (lambda_, degree) or (lambda_, gamma)
        """
        lambda_ = float(params[0])
        if model == "PolynomialKernelRidgeRegressor":
            degree = int(params[1])
        elif model == "RbfKernelRidgeRegressor":
            gamma = float(params[1])

        model_mapping = {
            "RidgeRegressor": lambda: RidgeRegressor(lambda_=lambda_),
            "LinearKernelRidgeRegressor": lambda: LinearKernelRidgeRegressor(lambda_=lambda_),
            "PolynomialKernelRidgeRegressor": lambda: PolynomialKernelRidgeRegressor(lambda_=lambda_, degree=degree),
            "RbfKernelRidgeRegressor": lambda: RbfKernelRidgeRegressor(lambda_=lambda_, gamma=gamma)
        }

        model_creator = model_mapping.get(model)
        if model_creator:
            self.ml_model = model_creator()
            self.model_creation_success()
        else:
            self.model_creation_success()
            raise ValueError(f"Internal Error: Invalid model type: {model}!")

    def model_creation_success(self):
        """Check if the model was created successfully.

        Emits appropriate signal based on model creation success.
        """
        if self.ml_model is None:
            self.model_creation_success_signal.emit(False, "")
        else:
            params_str = f"lambda_={self.ml_model.lambda_}"
            if isinstance(self.ml_model, RidgeRegressor):
                model_str = f"RidgeRegressor({params_str})"
            elif isinstance(self.ml_model, LinearKernelRidgeRegressor):
                model_str = f"LinearKernelRidgeRegressor({params_str})"
            elif isinstance(self.ml_model, PolynomialKernelRidgeRegressor):
                model_str = f"PolynomialKernelRidgeRegressor({params_str}, degree={self.ml_model.degree})"
            elif isinstance(self.ml_model, RbfKernelRidgeRegressor):
                model_str = f"RbfKernelRidgeRegressor({params_str}, gamma={self.ml_model.gamma})"
            else:
                self.model_creation_success_signal.emit(False, "")
                raise ValueError(f"Internal Error: Unable to create {self.ml_model.kernel_type}!")

            self.model_creation_success_signal.emit(True, model_str)

    @pyqtSlot()
    def fit_model(self) -> None:
        """Fit the model.

        Checks if the data has been preprocessed and fits the model to the data. Favours preprocessed data if available.
        """
        try:
            if self.data_scaled["X_train"] is None or self.data_scaled["y_train"] is None:
                self.ml_model.fit(self.data_split["X_train"], self.data_split["y_train"])
            else:
                self.ml_model.fit(self.data_scaled["X_train"], self.data_scaled["y_train"])
        except Exception as e:
            self.model_fit_success_signal.emit(False)
            print(e)
            raise ValueError(f"Error fitting model: {e}")
        print(self.ml_model.coef_)
        self.model_fit_success_signal.emit(True)

    @pyqtSlot()
    def get_predictions(self) -> None:
        """Get the predictions.

        Checks if the data has been preprocessed and gets the predictions. Favours preprocessed data if available.
        """
        if isinstance(self.ml_model, RidgeRegressor):
            try:
                if self.data_scaled["X_test"] is None:  # No preprocessing
                    y = self.data_split["y_test"]
                    y_pred = self.ml_model.predict(self.data_split["X_test"])
                else:  # Preprocessing
                    y = self.data_scaled["y_test"]
                    y_pred = self.ml_model.predict(self.data_scaled["X_test"])
            except Exception as e:
                print(e)
                raise ValueError(f"Error getting predictions: {e}")
            print(y_pred)
            self.predictions = y_pred
            self.send_predictions_signal.emit(y, y_pred)
        elif isinstance(self.ml_model, KernelRidgeRegressor):
            try:
                if self.data_scaled["X_test"] is None:  # No preprocessing
                    y = self.data_split["y_test"]
                    y_pred = self.ml_model.predict(self.data_split["X_test"])
                else:  # Preprocessing
                    y = self.data_scaled["y_test"]
                    y_pred = self.ml_model.predict(self.data_scaled["X_test"])
            except Exception as e:
                print(e)
                raise ValueError(f"Error getting predictions: {e}")
            print(y_pred)
            self.predictions = y_pred
            self.send_predictions_signal.emit(y, y_pred)

    @pyqtSlot()
    def evaluate_model(self) -> None:
        """Evaluate the model.
        
        Checks if the data has been preprocessed and evaluates the model. Favours preprocessed data if available.
        """
        try:
            if self.data_scaled["X_test"] is None:
                evals = self.ml_model.evaluate(self.data_split["y_test"], self.predictions)
            else:
                evals = self.ml_model.evaluate(self.data_scaled["y_test"], self.predictions)
        except Exception as e:
            print(e)
            raise ValueError(f"Error evaluating model: {e}")
        print(evals)
        if isinstance(self.ml_model, RidgeRegressor):
            model = f"{self.ml_model.__class__.__name__}(lambda_={self.ml_model.lambda_})"
        elif isinstance(self.ml_model, LinearKernelRidgeRegressor):
            model = f"{self.ml_model.__class__.__name__}(lambda_={self.ml_model.lambda_})"
        elif isinstance(self.ml_model, PolynomialKernelRidgeRegressor):
            model = f"{self.ml_model.__class__.__name__}(lambda_={self.ml_model.lambda_}, degree={self.ml_model.degree})"
        elif isinstance(self.ml_model, RbfKernelRidgeRegressor):
            model = f"{self.ml_model.__class__.__name__}(lambda_={self.ml_model.lambda_}, gamma={self.ml_model.gamma})"
        self.send_eval_res_signal.emit(model, evals)

    @pyqtSlot(int)
    def run_kfold(self, k: int):
        """Run k-fold cross validation.
        
        Args:
            k (int): the number of folds
        """
        k_fold = KFold(n_splits=k)
        try:
            if self.data_scaled["X_train"] is None:
                res = k_fold.validate(self.ml_model, self.data_split["X_train"], self.data_split["y_train"])
            else:
                res = k_fold.validate(self.ml_model, self.data_scaled["X_train"], self.data_scaled["y_train"])
        except Exception as e:
            print(e)
            raise ValueError(f"Error running k-fold cross validation: {e}")

        self.send_kfold_res_signal.emit(res)  # (scores, avg)
